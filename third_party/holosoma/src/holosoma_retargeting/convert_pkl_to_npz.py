#!/usr/bin/env python3
"""
将 .pkl 文件转换为项目使用的 .npz 格式
"""
import pickle
import numpy as np
import mujoco
from pathlib import Path
import sys

def load_robot_model(xml_path):
    """加载机器人XML模型（MuJoCo格式）"""
    # 使用绝对路径或相对于脚本目录的路径
    xml_path = Path(xml_path)
    if not xml_path.is_absolute():
        # 如果是相对路径，尝试从脚本目录查找
        script_dir = Path(__file__).parent
        xml_path = script_dir / xml_path
        if not xml_path.exists():
            # 尝试从models目录查找
            xml_path = script_dir / "models" / "g1" / "g1_29dof.xml"
    
    if not xml_path.exists():
        raise FileNotFoundError(f"找不到XML文件: {xml_path}")
    
    # 使用from_xml_path会自动处理相对路径的mesh文件
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    return model, data

def get_body_names(model):
    """获取所有body名称"""
    body_names = []
    for i in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
        if name:
            body_names.append(name)
    return body_names

def get_joint_names(model):
    """获取所有关节名称（不包括root）"""
    joint_names = []
    for i in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        if name and name != "root":  # 排除root joint
            joint_names.append(name)
    return joint_names

def compute_body_positions(model, data, qpos):
    """
    使用正向运动学计算所有body的位置和旋转
    
    Args:
        model: MuJoCo model
        data: MuJoCo data
        qpos: (7 + ndof,) 或 (T, 7 + ndof) 关节位置 [root_pos(3), root_quat(4), joints(ndof)]
    
    Returns:
        body_pos_w: (T, nbody, 3) 或 (nbody, 3)
        body_quat_w: (T, nbody, 4) 或 (nbody, 4)
    """
    is_batch = qpos.ndim == 2
    if not is_batch:
        qpos = qpos[None, :]
    
    T = qpos.shape[0]
    nbody = model.nbody
    
    body_pos_w = np.zeros((T, nbody, 3))
    body_quat_w = np.zeros((T, nbody, 4))
    
    for t in range(T):
        # 设置关节位置
        data.qpos[:] = qpos[t]
        # 正向运动学
        mujoco.mj_forward(model, data)
        
        # 获取所有body的位置和旋转
        for i in range(nbody):
            body_pos_w[t, i] = data.xpos[i].copy()
            # MuJoCo使用四元数 (w,x,y,z)，需要转换为 (w,x,y,z)
            quat = data.xquat[i].copy()
            body_quat_w[t, i] = quat  # MuJoCo已经是(w,x,y,z)格式
    
    if not is_batch:
        body_pos_w = body_pos_w[0]
        body_quat_w = body_quat_w[0]
    
    return body_pos_w, body_quat_w

def compute_velocities(positions, quaternions, fps):
    """
    通过差分计算速度
    
    Args:
        positions: (T, n, 3)
        quaternions: (T, n, 4)
        fps: 帧率
    
    Returns:
        lin_vel: (T, n, 3)
        ang_vel: (T, n, 3) - 角速度（从四元数差分计算）
    """
    dt = 1.0 / fps
    T = positions.shape[0]
    
    lin_vel = np.zeros_like(positions)
    ang_vel = np.zeros((T, positions.shape[1], 3))
    
    # 线性速度
    lin_vel[1:] = (positions[1:] - positions[:-1]) / dt
    lin_vel[0] = lin_vel[1]  # 第一帧使用第二帧的速度
    
    # 角速度（简化：从四元数差分计算）
    for i in range(1, T):
        for j in range(quaternions.shape[1]):
            q0 = quaternions[i-1, j]
            q1 = quaternions[i, j]
            # 简化的角速度计算（只取xyz部分）
            dq = q1[:3] - q0[:3]
            ang_vel[i, j] = dq / dt
    ang_vel[0] = ang_vel[1]
    
    return lin_vel, ang_vel

def convert_pkl_to_npz(pkl_path, output_path, xml_path, fps=None):
    """
    将 .pkl 文件转换为 .npz 格式
    
    Args:
        pkl_path: 输入的 .pkl 文件路径
        output_path: 输出的 .npz 文件路径
        urdf_path: 机器人URDF文件路径
        fps: 帧率（如果pkl中没有）
    """
    print(f"加载 .pkl 文件: {pkl_path}")
    with open(pkl_path, 'rb') as f:
        pkl_data = pickle.load(f)
    
    # 提取数据
    root_pos = pkl_data['root_pos']  # (T, 3)
    root_rot = pkl_data['root_rot']  # (T, 4) - 四元数
    dof_pos = pkl_data['dof_pos']    # (T, 29)
    pkl_fps = pkl_data.get('fps', fps) if fps is None else fps
    
    if pkl_fps is None:
        pkl_fps = 30.0  # 默认30fps
    
    T = root_pos.shape[0]
    ndof = dof_pos.shape[1]
    
    print(f"数据形状: T={T}, ndof={ndof}, fps={pkl_fps}")
    print(f"root_pos shape: {root_pos.shape}")
    print(f"root_rot shape: {root_rot.shape}")
    print(f"dof_pos shape: {dof_pos.shape}")
    
    # 检查四元数格式（可能是xyzw或wxyz）
    # 假设是wxyz格式（与项目一致）
    root_quat = root_rot.copy()  # (T, 4)
    
    # 构建 joint_pos: [root_pos(3), root_quat(4), dof_pos(ndof)]
    joint_pos = np.zeros((T, 7 + ndof))
    joint_pos[:, 0:3] = root_pos
    joint_pos[:, 3:7] = root_quat
    joint_pos[:, 7:] = dof_pos
    
    print(f"加载机器人模型: {xml_path}")
    model, data = load_robot_model(xml_path)
    
    # 获取body和joint名称
    body_names = get_body_names(model)
    joint_names = get_joint_names(model)
    
    print(f"机器人body数量: {len(body_names)}")
    print(f"机器人joint数量: {len(joint_names)}")
    
    # 计算body位置和旋转
    print("计算body位置和旋转...")
    body_pos_w, body_quat_w = compute_body_positions(model, data, joint_pos)
    
    # 计算速度
    print("计算速度...")
    # joint_vel: [root_lin_vel(3), root_ang_vel(3), joint_vel(ndof)]
    joint_vel = np.zeros((T, 6 + ndof))
    joint_vel[:, 0:3] = np.gradient(root_pos, axis=0) * pkl_fps  # root线性速度
    # root角速度（从四元数差分计算，简化）
    root_ang_vel = np.gradient(root_quat[:, 1:], axis=0) * pkl_fps  # 简化：只用xyz部分
    joint_vel[:, 3:6] = root_ang_vel
    joint_vel[:, 6:] = np.gradient(dof_pos, axis=0) * pkl_fps  # 关节速度
    
    # body速度
    body_lin_vel_w, body_ang_vel_w = compute_velocities(body_pos_w, body_quat_w, pkl_fps)
    
    # 保存为 .npz
    print(f"保存到: {output_path}")
    np.savez(
        output_path,
        joint_pos=joint_pos.astype(np.float32),
        joint_vel=joint_vel.astype(np.float32),
        body_pos_w=body_pos_w.astype(np.float32),
        body_quat_w=body_quat_w.astype(np.float32),
        body_lin_vel_w=body_lin_vel_w.astype(np.float32),
        body_ang_vel_w=body_ang_vel_w.astype(np.float32),
        joint_names=np.array(joint_names),
        body_names=np.array(body_names),
        fps=np.array([pkl_fps])
    )
    
    print(f"✓ 转换完成！")
    print(f"  总帧数: {T}")
    print(f"  帧率: {pkl_fps} fps")
    print(f"  时长: {T/pkl_fps:.2f} 秒")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="将 .pkl 文件转换为 .npz 格式")
    parser.add_argument("--pkl_path", type=str, required=True, help="输入的 .pkl 文件路径")
    parser.add_argument("--output_path", type=str, required=True, help="输出的 .npz 文件路径")
    parser.add_argument("--xml_path", type=str, default="models/g1/g1_29dof.xml", help="机器人XML文件路径（MuJoCo格式）")
    parser.add_argument("--fps", type=float, default=None, help="帧率（如果pkl中没有）")
    
    args = parser.parse_args()
    
    convert_pkl_to_npz(args.pkl_path, args.output_path, args.xml_path, args.fps)
