#!/usr/bin/env python3
"""
在motion文件的开头和结尾都添加default pose插值
"""
import numpy as np

def quat_slerp(q0, q1, t):
    """
    Spherical linear interpolation for quaternions (wxyz format)
    q0, q1: (4,) quaternions in wxyz format
    t: scalar or array of interpolation factors [0, 1]
    Returns: interpolated quaternion(s)
    """
    t = np.asarray(t)
    if t.ndim == 0:
        t = t.reshape(1)
    
    # Normalize quaternions
    q0_norm = np.linalg.norm(q0)
    q1_norm = np.linalg.norm(q1)
    if q0_norm > 1e-8:
        q0 = q0 / q0_norm
    if q1_norm > 1e-8:
        q1 = q1 / q1_norm
    
    # Compute dot product
    dot = np.clip(np.dot(q0, q1), -1.0, 1.0)
    
    # If quaternions are very close, use linear interpolation
    if abs(dot) > 0.9995:
        result = q0 + t[:, None] * (q1 - q0)
        return result / np.linalg.norm(result, axis=1, keepdims=True)
    
    # Calculate angle
    theta = np.arccos(abs(dot))
    sin_theta = np.sin(theta)
    
    # Interpolate
    w0 = np.sin((1 - t) * theta) / sin_theta
    w1 = np.sin(t * theta) / sin_theta
    
    if dot < 0:
        q1 = -q1
    
    result = w0[:, None] * q0 + w1[:, None] * q1
    return result / np.linalg.norm(result, axis=1, keepdims=True)

def add_interpolation_both_ends(default_pose_file, input_file, output_file, num_interp_frames=13):
    """
    在输入文件的开头和结尾都添加default pose插值
    
    Args:
        default_pose_file: default pose文件（单帧）
        input_file: 输入motion文件
        output_file: 输出文件（添加插值后的文件）
        num_interp_frames: 每端插值帧数（默认13帧，50fps约0.25秒）
    """
    print(f"加载default pose文件: {default_pose_file}")
    default_data = np.load(default_pose_file, allow_pickle=True)
    
    print(f"加载输入文件: {input_file}")
    input_data = np.load(input_file, allow_pickle=True)
    
    # 验证文件格式
    assert list(default_data.keys()) == list(input_data.keys()), "文件格式不一致"
    
    fps = float(np.array(input_data["fps"]).reshape(-1)[0])
    print(f"FPS: {fps}")
    print(f"每端插值帧数: {num_interp_frames} ({num_interp_frames/fps:.2f}秒)")
    
    # 提取default pose（第0帧）
    default_joint_pos = default_data["joint_pos"][0]  # (7+ndof,)
    default_joint_vel = default_data["joint_vel"][0]  # (6+ndof,)
    default_body_pos_w = default_data["body_pos_w"][0]  # (nbody, 3)
    default_body_quat_w = default_data["body_quat_w"][0]  # (nbody, 4) wxyz
    default_body_lin_vel_w = default_data["body_lin_vel_w"][0]  # (nbody, 3)
    default_body_ang_vel_w = default_data["body_ang_vel_w"][0]  # (nbody, 3)
    
    # 提取输入文件的第一帧和最后一帧
    first_joint_pos = input_data["joint_pos"][0]  # (7+ndof,)
    first_joint_vel = input_data["joint_vel"][0]  # (6+ndof,)
    first_body_pos_w = input_data["body_pos_w"][0]  # (nbody, 3)
    first_body_quat_w = input_data["body_quat_w"][0]  # (nbody, 4) wxyz
    first_body_lin_vel_w = input_data["body_lin_vel_w"][0]  # (nbody, 3)
    first_body_ang_vel_w = input_data["body_ang_vel_w"][0]  # (nbody, 3)
    
    last_joint_pos = input_data["joint_pos"][-1]  # (7+ndof,)
    last_joint_vel = input_data["joint_vel"][-1]  # (6+ndof,)
    last_body_pos_w = input_data["body_pos_w"][-1]  # (nbody, 3)
    last_body_quat_w = input_data["body_quat_w"][-1]  # (nbody, 4) wxyz
    last_body_lin_vel_w = input_data["body_lin_vel_w"][-1]  # (nbody, 3)
    last_body_ang_vel_w = input_data["body_ang_vel_w"][-1]  # (nbody, 3)
    
    # 插值参数
    alphas = np.linspace(0, 1, num_interp_frames)
    
    # ========== 开头插值：从default pose到第一帧 ==========
    print("计算开头插值...")
    interp_start_joint_pos = np.zeros((num_interp_frames, default_joint_pos.shape[0]))
    interp_start_joint_vel = np.zeros((num_interp_frames, default_joint_vel.shape[0]))
    interp_start_body_pos_w = np.zeros((num_interp_frames, default_body_pos_w.shape[0], 3))
    interp_start_body_quat_w = np.zeros((num_interp_frames, default_body_quat_w.shape[0], 4))
    interp_start_body_lin_vel_w = np.zeros((num_interp_frames, default_body_lin_vel_w.shape[0], 3))
    interp_start_body_ang_vel_w = np.zeros((num_interp_frames, default_body_ang_vel_w.shape[0], 3))
    
    for i, alpha in enumerate(alphas):
        # joint_pos插值
        # 修复：保持x,y位置为第一帧位置，只插值z位置
        target_xy = first_joint_pos[:2]  # 使用第一帧的x,y
        default_z = default_joint_pos[2]
        first_z = first_joint_pos[2]
        interp_z = default_z + alpha * (first_z - default_z)  # 只插值z
        interp_start_joint_pos[i, :3] = np.array([target_xy[0], target_xy[1], interp_z])
        
        # root quat (wxyz) - 使用slerp
        interp_start_joint_pos[i, 3:7] = quat_slerp(default_joint_pos[3:7], first_joint_pos[3:7], alpha)[0]
        # joint angles - 线性插值
        interp_start_joint_pos[i, 7:] = default_joint_pos[7:] + alpha * (first_joint_pos[7:] - default_joint_pos[7:])
        
        # joint_vel插值
        interp_start_joint_vel[i, :3] = default_joint_vel[:3] + alpha * (first_joint_vel[:3] - default_joint_vel[:3])
        interp_start_joint_vel[i, 3:6] = default_joint_vel[3:6] + alpha * (first_joint_vel[3:6] - default_joint_vel[3:6])
        interp_start_joint_vel[i, 6:] = default_joint_vel[6:] + alpha * (first_joint_vel[6:] - default_joint_vel[6:])
        
        # body插值
        for j in range(default_body_pos_w.shape[0]):
            # 保持x,y位置为第一帧位置，只插值z位置
            target_xy = first_body_pos_w[j, :2]
            default_z = default_body_pos_w[j, 2]
            first_z = first_body_pos_w[j, 2]
            interp_z = default_z + alpha * (first_z - default_z)
            interp_start_body_pos_w[i, j] = np.array([target_xy[0], target_xy[1], interp_z])
            
            # body quat - 使用slerp
            interp_start_body_quat_w[i, j] = quat_slerp(default_body_quat_w[j], first_body_quat_w[j], alpha)[0]
            
            # body速度 - 线性插值
            interp_start_body_lin_vel_w[i, j] = default_body_lin_vel_w[j] + alpha * (first_body_lin_vel_w[j] - default_body_lin_vel_w[j])
            interp_start_body_ang_vel_w[i, j] = default_body_ang_vel_w[j] + alpha * (first_body_ang_vel_w[j] - default_body_ang_vel_w[j])
    
    # ========== 结尾插值：从最后一帧到default pose ==========
    print("计算结尾插值...")
    interp_end_joint_pos = np.zeros((num_interp_frames, last_joint_pos.shape[0]))
    interp_end_joint_vel = np.zeros((num_interp_frames, last_joint_vel.shape[0]))
    interp_end_body_pos_w = np.zeros((num_interp_frames, last_body_pos_w.shape[0], 3))
    interp_end_body_quat_w = np.zeros((num_interp_frames, last_body_quat_w.shape[0], 4))
    interp_end_body_lin_vel_w = np.zeros((num_interp_frames, last_body_lin_vel_w.shape[0], 3))
    interp_end_body_ang_vel_w = np.zeros((num_interp_frames, last_body_ang_vel_w.shape[0], 3))
    
    for i, alpha in enumerate(alphas):
        # joint_pos插值：从最后一帧到default pose
        # 保持x,y位置为最后一帧位置，只插值z位置
        target_xy = last_joint_pos[:2]  # 使用最后一帧的x,y
        last_z = last_joint_pos[2]
        default_z = default_joint_pos[2]
        interp_z = last_z + alpha * (default_z - last_z)  # 从最后一帧z插值到default z
        interp_end_joint_pos[i, :3] = np.array([target_xy[0], target_xy[1], interp_z])
        
        # root quat (wxyz) - 使用slerp
        interp_end_joint_pos[i, 3:7] = quat_slerp(last_joint_pos[3:7], default_joint_pos[3:7], alpha)[0]
        # joint angles - 线性插值
        interp_end_joint_pos[i, 7:] = last_joint_pos[7:] + alpha * (default_joint_pos[7:] - last_joint_pos[7:])
        
        # joint_vel插值
        interp_end_joint_vel[i, :3] = last_joint_vel[:3] + alpha * (default_joint_vel[:3] - last_joint_vel[:3])
        interp_end_joint_vel[i, 3:6] = last_joint_vel[3:6] + alpha * (default_joint_vel[3:6] - last_joint_vel[3:6])
        interp_end_joint_vel[i, 6:] = last_joint_vel[6:] + alpha * (default_joint_vel[6:] - last_joint_vel[6:])
        
        # body插值
        for j in range(last_body_pos_w.shape[0]):
            # 保持x,y位置为最后一帧位置，只插值z位置
            target_xy = last_body_pos_w[j, :2]
            last_z = last_body_pos_w[j, 2]
            default_z = default_body_pos_w[j, 2]
            interp_z = last_z + alpha * (default_z - last_z)
            interp_end_body_pos_w[i, j] = np.array([target_xy[0], target_xy[1], interp_z])
            
            # body quat - 使用slerp
            interp_end_body_quat_w[i, j] = quat_slerp(last_body_quat_w[j], default_body_quat_w[j], alpha)[0]
            
            # body速度 - 线性插值
            interp_end_body_lin_vel_w[i, j] = last_body_lin_vel_w[j] + alpha * (default_body_lin_vel_w[j] - last_body_lin_vel_w[j])
            interp_end_body_ang_vel_w[i, j] = last_body_ang_vel_w[j] + alpha * (default_body_ang_vel_w[j] - last_body_ang_vel_w[j])
    
    # ========== 合并：开头插值 + 原始数据 + 结尾插值 ==========
    print("合并数据...")
    output_data = {}
    for key in input_data.keys():
        if key in ['joint_names', 'body_names', 'fps']:
            output_data[key] = input_data[key]
        elif key == 'joint_pos':
            output_data[key] = np.concatenate([
                interp_start_joint_pos,
                input_data[key],
                interp_end_joint_pos
            ], axis=0)
        elif key == 'joint_vel':
            output_data[key] = np.concatenate([
                interp_start_joint_vel,
                input_data[key],
                interp_end_joint_vel
            ], axis=0)
        elif key == 'body_pos_w':
            output_data[key] = np.concatenate([
                interp_start_body_pos_w,
                input_data[key],
                interp_end_body_pos_w
            ], axis=0)
        elif key == 'body_quat_w':
            output_data[key] = np.concatenate([
                interp_start_body_quat_w,
                input_data[key],
                interp_end_body_quat_w
            ], axis=0)
        elif key == 'body_lin_vel_w':
            output_data[key] = np.concatenate([
                interp_start_body_lin_vel_w,
                input_data[key],
                interp_end_body_lin_vel_w
            ], axis=0)
        elif key == 'body_ang_vel_w':
            output_data[key] = np.concatenate([
                interp_start_body_ang_vel_w,
                input_data[key],
                interp_end_body_ang_vel_w
            ], axis=0)
        else:
            # 其他字段直接复制
            output_data[key] = input_data[key]
    
    # 保存
    print(f"\n保存到: {output_file}")
    np.savez(output_file, **output_data)
    
    original_frames = input_data['joint_pos'].shape[0]
    new_frames = output_data['joint_pos'].shape[0]
    print(f"✓ 完成！")
    print(f"  原始帧数: {original_frames}")
    print(f"  开头插值: {num_interp_frames} 帧")
    print(f"  结尾插值: {num_interp_frames} 帧")
    print(f"  总帧数: {new_frames}")
    print(f"  总时长: {new_frames/fps:.2f}秒")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="在motion文件的开头和结尾都添加default pose插值")
    parser.add_argument("--input_file", type=str, required=True, help="输入的npz文件路径")
    parser.add_argument("--output_file", type=str, required=True, help="输出的npz文件路径")
    parser.add_argument("--default_pose_file", type=str, required=True, help="default pose的npz文件路径")
    parser.add_argument("--interp_duration", type=float, default=0.25, help="插值时长（秒，默认0.25）")
    
    args = parser.parse_args()
    
    # 根据fps计算插值需要的帧数
    input_data = np.load(args.input_file, allow_pickle=True)
    fps = float(np.array(input_data["fps"]).reshape(-1)[0])
    num_frames = int(fps * args.interp_duration)
    print(f"FPS: {fps}, {args.interp_duration}秒 = {num_frames}帧")
    
    add_interpolation_both_ends(
        default_pose_file=args.default_pose_file,
        input_file=args.input_file,
        output_file=args.output_file,
        num_interp_frames=num_frames
    )
