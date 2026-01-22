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

def smooth_velocity(velocity, window_size=3):
    """
    使用移动平均平滑速度
    
    Args:
        velocity: (T, ...) 速度数组
        window_size: 移动平均窗口大小（必须是奇数）
    
    Returns:
        smoothed_velocity: (T, ...) 平滑后的速度
    """
    if window_size % 2 == 0:
        window_size += 1  # 确保是奇数
    
    T = velocity.shape[0]
    if T < window_size:
        return velocity
    
    # 使用卷积进行移动平均
    kernel = np.ones(window_size) / window_size
    smoothed = np.zeros_like(velocity)
    
    # 处理每个维度
    if velocity.ndim == 1:
        smoothed = np.convolve(velocity, kernel, mode='same')
    elif velocity.ndim == 2:
        for i in range(velocity.shape[1]):
            smoothed[:, i] = np.convolve(velocity[:, i], kernel, mode='same')
    elif velocity.ndim == 3:
        for i in range(velocity.shape[1]):
            for j in range(velocity.shape[2]):
                smoothed[:, i, j] = np.convolve(velocity[:, i, j], kernel, mode='same')
    else:
        # 对于更高维度，使用reshape
        orig_shape = velocity.shape
        velocity_flat = velocity.reshape(T, -1)
        smoothed_flat = np.zeros_like(velocity_flat)
        for i in range(velocity_flat.shape[1]):
            smoothed_flat[:, i] = np.convolve(velocity_flat[:, i], kernel, mode='same')
        smoothed = smoothed_flat.reshape(orig_shape)
    
    return smoothed

def compute_velocities(positions, quaternions, fps, smooth=True, smooth_window=5):
    """
    通过差分计算速度（添加NaN保护和速度平滑）
    
    Args:
        positions: (T, n, 3)
        quaternions: (T, n, 4)
        fps: 帧率
        smooth: 是否平滑速度
        smooth_window: 平滑窗口大小
    
    Returns:
        lin_vel: (T, n, 3)
        ang_vel: (T, n, 3) - 角速度（从四元数差分计算）
    """
    dt = 1.0 / fps
    T = positions.shape[0]
    
    # 检查输入是否有NaN/Inf
    if np.isnan(positions).any():
        print("警告: positions包含NaN，将被替换为0")
        positions = np.nan_to_num(positions, nan=0.0)
    if np.isinf(positions).any():
        print("警告: positions包含Inf，将被替换为0")
        positions = np.nan_to_num(positions, posinf=0.0, neginf=0.0)
    
    if np.isnan(quaternions).any():
        print("警告: quaternions包含NaN，将被替换为单位四元数")
        quaternions = quaternions.copy()
        # 确保四元数归一化
        for i in range(T):
            for j in range(quaternions.shape[1]):
                q = quaternions[i, j]
                if np.isnan(q).any() or np.isinf(q).any():
                    quaternions[i, j] = np.array([1.0, 0.0, 0.0, 0.0])  # 默认单位四元数
                else:
                    norm = np.linalg.norm(q)
                    if norm > 1e-8:
                        quaternions[i, j] = q / norm
                    else:
                        quaternions[i, j] = np.array([1.0, 0.0, 0.0, 0.0])
    
    lin_vel = np.zeros_like(positions)
    ang_vel = np.zeros((T, positions.shape[1], 3))
    
    # 线性速度
    if T > 1:
        lin_vel[1:] = (positions[1:] - positions[:-1]) / dt
        lin_vel[0] = lin_vel[1]  # 第一帧使用第二帧的速度
    else:
        lin_vel[0] = 0.0
    
    # 检查速度中的NaN/Inf
    if np.isnan(lin_vel).any() or np.isinf(lin_vel).any():
        print("警告: 计算出的线性速度包含NaN/Inf，将被替换为0")
        lin_vel = np.nan_to_num(lin_vel, nan=0.0, posinf=0.0, neginf=0.0)
    
    # 角速度（简化：从四元数差分计算）
    if T > 1:
        for i in range(1, T):
            for j in range(quaternions.shape[1]):
                q0 = quaternions[i-1, j]
                q1 = quaternions[i, j]
                # 简化的角速度计算（只取xyz部分）
                dq = q1[:3] - q0[:3]
                ang_vel[i, j] = dq / dt
        ang_vel[0] = ang_vel[1]
    else:
        ang_vel[0] = 0.0
    
    # 检查角速度中的NaN/Inf
    if np.isnan(ang_vel).any() or np.isinf(ang_vel).any():
        print("警告: 计算出的角速度包含NaN/Inf，将被替换为0")
        ang_vel = np.nan_to_num(ang_vel, nan=0.0, posinf=0.0, neginf=0.0)
    
    # 平滑速度（如果启用）
    if smooth and T > smooth_window:
        print(f"平滑速度（窗口大小={smooth_window}）...")
        lin_vel = smooth_velocity(lin_vel, window_size=smooth_window)
        ang_vel = smooth_velocity(ang_vel, window_size=smooth_window)
        print(f"  平滑后线性速度范围: [{lin_vel.min():.2f}, {lin_vel.max():.2f}]")
        print(f"  平滑后角速度范围: [{ang_vel.min():.2f}, {ang_vel.max():.2f}]")
    
    return lin_vel, ang_vel

def smooth_joint_velocity(joint_vel, window_size=5):
    """
    平滑关节速度（包括root和dof）
    
    Args:
        joint_vel: (T, 6+ndof) 关节速度
        window_size: 平滑窗口大小
    
    Returns:
        smoothed_velocity: (T, 6+ndof) 平滑后的速度
    """
    return smooth_velocity(joint_vel, window_size=window_size)

def convert_pkl_to_npz(pkl_path, output_path, xml_path, fps=None, smooth_velocities=True, smooth_window=5, velocity_scale=None):
    """
    将 .pkl 文件转换为 .npz 格式
    
    Args:
        pkl_path: 输入的 .pkl 文件路径
        output_path: 输出的 .npz 文件路径
        xml_path: 机器人XML文件路径
        fps: 帧率（如果pkl中没有）
        smooth_velocities: 是否平滑速度
        smooth_window: 平滑窗口大小
        velocity_scale: 速度缩放因子（如果指定，会将所有速度乘以该因子）
    """
    print(f"加载 .pkl 文件: {pkl_path}")
    with open(pkl_path, 'rb') as f:
        pkl_data = pickle.load(f)
    
    # 提取数据
    root_pos = pkl_data['root_pos']  # (T, 3)
    root_rot = pkl_data['root_rot']  # (T, 4) - 四元数
    dof_pos = pkl_data['dof_pos']    # (T, 29)
    input_fps = pkl_data.get('fps', fps) if fps is None else pkl_data.get('fps', 30.0)
    output_fps = fps if fps is not None else input_fps  # 输出FPS，如果未指定则使用输入FPS
    
    if input_fps is None:
        input_fps = 30.0  # 默认30fps
    if output_fps is None:
        output_fps = input_fps
    
    T = root_pos.shape[0]
    ndof = dof_pos.shape[1]
    
    print(f"数据形状: T={T}, ndof={ndof}")
    print(f"输入FPS: {input_fps}, 输出FPS: {output_fps}")
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
    # 重要：速度计算应该使用输出FPS，而不是输入FPS
    # 因为速度 = dx/dt，而dt应该基于输出FPS
    print("计算速度...")
    print(f"  使用输出FPS ({output_fps}) 计算速度（而不是输入FPS {input_fps}）")
    # joint_vel: [root_lin_vel(3), root_ang_vel(3), joint_vel(ndof)]
    joint_vel = np.zeros((T, 6 + ndof))
    joint_vel[:, 0:3] = np.gradient(root_pos, axis=0) * output_fps  # root线性速度
    # root角速度（从四元数差分计算，简化）
    root_ang_vel = np.gradient(root_quat[:, 1:], axis=0) * output_fps  # 简化：只用xyz部分
    joint_vel[:, 3:6] = root_ang_vel
    joint_vel[:, 6:] = np.gradient(dof_pos, axis=0) * output_fps  # 关节速度
    
    # 平滑joint速度（如果启用）
    if smooth_velocities and T > smooth_window:
        print(f"平滑joint速度（窗口大小={smooth_window}）...")
        print(f"  平滑前joint_vel范围: [{joint_vel.min():.2f}, {joint_vel.max():.2f}]")
        joint_vel = smooth_joint_velocity(joint_vel, window_size=smooth_window)
        print(f"  平滑后joint_vel范围: [{joint_vel.min():.2f}, {joint_vel.max():.2f}]")
    
    # body速度（使用输出FPS）
    body_lin_vel_w, body_ang_vel_w = compute_velocities(body_pos_w, body_quat_w, output_fps, smooth=smooth_velocities, smooth_window=smooth_window)
    
    # 速度缩放（如果指定）
    if velocity_scale is not None:
        print(f"\n缩放速度（缩放因子={velocity_scale:.3f}）...")
        print(f"  缩放前joint_vel范围: [{joint_vel.min():.2f}, {joint_vel.max():.2f}]")
        joint_vel = joint_vel * velocity_scale
        body_lin_vel_w = body_lin_vel_w * velocity_scale
        body_ang_vel_w = body_ang_vel_w * velocity_scale
        print(f"  缩放后joint_vel范围: [{joint_vel.min():.2f}, {joint_vel.max():.2f}]")
        print(f"  缩放后body_lin_vel_w范围: [{body_lin_vel_w.min():.2f}, {body_lin_vel_w.max():.2f}]")
        print(f"  缩放后body_ang_vel_w范围: [{body_ang_vel_w.min():.2f}, {body_ang_vel_w.max():.2f}]")
    
    # 最终验证：检查所有输出数据
    print("\n验证输出数据...")
    all_clean = True
    data_to_save = {
        'joint_pos': joint_pos,
        'joint_vel': joint_vel,
        'body_pos_w': body_pos_w,
        'body_quat_w': body_quat_w,
        'body_lin_vel_w': body_lin_vel_w,
        'body_ang_vel_w': body_ang_vel_w,
    }
    
    for key, arr in data_to_save.items():
        nan_count = np.isnan(arr).sum()
        inf_count = np.isinf(arr).sum()
        if nan_count > 0 or inf_count > 0:
            print(f"⚠️  {key}: 发现 {nan_count} 个NaN, {inf_count} 个Inf，正在清理...")
            data_to_save[key] = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
            all_clean = False
    
    if not all_clean:
        print("⚠️  数据已清理，但建议检查原始数据源")
    else:
        print("✓ 所有数据正常")
    
    # 保存为 .npz
    print(f"\n保存到: {output_path}")
    np.savez(
        output_path,
        joint_pos=data_to_save['joint_pos'].astype(np.float32),
        joint_vel=data_to_save['joint_vel'].astype(np.float32),
        body_pos_w=data_to_save['body_pos_w'].astype(np.float32),
        body_quat_w=data_to_save['body_quat_w'].astype(np.float32),
        body_lin_vel_w=data_to_save['body_lin_vel_w'].astype(np.float32),
        body_ang_vel_w=data_to_save['body_ang_vel_w'].astype(np.float32),
        joint_names=np.array(joint_names),
        body_names=np.array(body_names),
        fps=np.array([output_fps])  # 保存输出FPS
    )
    
    print(f"✓ 转换完成！")
    print(f"  总帧数: {T}")
    print(f"  帧率: {output_fps} fps")
    print(f"  时长: {T/output_fps:.2f} 秒")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="将 .pkl 文件转换为 .npz 格式")
    parser.add_argument("--pkl_path", type=str, required=True, help="输入的 .pkl 文件路径")
    parser.add_argument("--output_path", type=str, required=True, help="输出的 .npz 文件路径")
    parser.add_argument("--xml_path", type=str, default="models/g1/g1_29dof.xml", help="机器人XML文件路径（MuJoCo格式）")
    parser.add_argument("--fps", type=float, default=None, help="帧率（如果pkl中没有）")
    parser.add_argument("--smooth_velocities", action="store_true", default=True, help="是否平滑速度（默认启用）")
    parser.add_argument("--no_smooth_velocities", dest="smooth_velocities", action="store_false", help="禁用速度平滑")
    parser.add_argument("--smooth_window", type=int, default=5, help="速度平滑窗口大小（默认5）")
    parser.add_argument("--velocity_scale", type=float, default=None, help="速度缩放因子（例如0.619将速度缩放到约1.5倍官方数据）")
    
    args = parser.parse_args()
    
    convert_pkl_to_npz(args.pkl_path, args.output_path, args.xml_path, args.fps, args.smooth_velocities, args.smooth_window, args.velocity_scale)
