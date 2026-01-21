#!/usr/bin/env python3
"""
从参考motion文件提取初始姿态，并在剪切后的文件前添加插值帧
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
    q0 = q0 / np.linalg.norm(q0)
    q1 = q1 / np.linalg.norm(q1)
    
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

def prepend_interpolation(reference_file, input_file, output_file, num_interp_frames=100, reference_frame_idx=0):
    """
    从参考文件提取初始姿态，在输入文件前添加插值帧
    
    Args:
        reference_file: 参考motion文件（从中提取初始姿态）
        input_file: 输入motion文件（剪切后的文件）
        output_file: 输出文件（添加插值后的文件）
        num_interp_frames: 插值帧数（默认100帧，50fps约2秒）
        reference_frame_idx: 从参考文件中提取第几帧（默认第0帧）
    """
    print(f"加载参考文件: {reference_file}")
    ref_data = np.load(reference_file, allow_pickle=True)
    
    print(f"加载输入文件: {input_file}")
    input_data = np.load(input_file, allow_pickle=True)
    
    # 验证文件格式
    assert list(ref_data.keys()) == list(input_data.keys()), "文件格式不一致"
    
    fps = float(np.array(ref_data["fps"]).reshape(-1)[0])
    print(f"FPS: {fps}")
    print(f"插值帧数: {num_interp_frames} ({num_interp_frames/fps:.2f}秒)")
    
    # 提取参考帧（初始姿态）
    ref_joint_pos = ref_data["joint_pos"][reference_frame_idx]  # (7+ndof,)
    ref_joint_vel = ref_data["joint_vel"][reference_frame_idx]  # (6+ndof,)
    ref_body_pos_w = ref_data["body_pos_w"][reference_frame_idx]  # (nbody, 3)
    ref_body_quat_w = ref_data["body_quat_w"][reference_frame_idx]  # (nbody, 4) wxyz
    ref_body_lin_vel_w = ref_data["body_lin_vel_w"][reference_frame_idx]  # (nbody, 3)
    ref_body_ang_vel_w = ref_data["body_ang_vel_w"][reference_frame_idx]  # (nbody, 3)
    
    # 提取目标帧（输入文件的第一帧）
    target_joint_pos = input_data["joint_pos"][0]  # (7+ndof,)
    target_joint_vel = input_data["joint_vel"][0]  # (6+ndof,)
    target_body_pos_w = input_data["body_pos_w"][0]  # (nbody, 3)
    target_body_quat_w = input_data["body_quat_w"][0]  # (nbody, 4) wxyz
    target_body_lin_vel_w = input_data["body_lin_vel_w"][0]  # (nbody, 3)
    target_body_ang_vel_w = input_data["body_ang_vel_w"][0]  # (nbody, 3)
    
    print(f"\n参考帧（第{reference_frame_idx}帧）:")
    print(f"  Root位置: {ref_joint_pos[:3]}")
    print(f"  关节角度范围: [{ref_joint_pos[7:].min():.3f}, {ref_joint_pos[7:].max():.3f}]")
    
    print(f"\n目标帧（输入文件第0帧）:")
    print(f"  Root位置: {target_joint_pos[:3]}")
    print(f"  关节角度范围: [{target_joint_pos[7:].min():.3f}, {target_joint_pos[7:].max():.3f}]")
    
    # 创建插值系数（不包括最后一帧，因为目标帧会保留）
    alphas = np.linspace(0, 1, num_interp_frames + 1)[:-1]  # [0, ..., 0.99]
    
    # 插值joint_pos
    # 修复：保持x,y位置为目标帧位置，避免水平漂移
    interp_joint_pos = np.zeros((num_interp_frames, ref_joint_pos.shape[0]))
    for i, alpha in enumerate(alphas):
        # Root位置：保持x,y为目标帧位置，只插值z（如果需要平滑过渡）
        target_xy = target_joint_pos[:2]  # 使用目标帧的x,y
        ref_z = ref_joint_pos[2]
        target_z = target_joint_pos[2]
        interp_z = ref_z + alpha * (target_z - ref_z)  # 只插值z
        interp_joint_pos[i, :3] = np.array([target_xy[0], target_xy[1], interp_z])
        
        # root quat (wxyz) - 使用slerp
        interp_joint_pos[i, 3:7] = quat_slerp(ref_joint_pos[3:7], target_joint_pos[3:7], alpha)[0]
        # joint angles - 线性插值
        interp_joint_pos[i, 7:] = ref_joint_pos[7:] + alpha * (target_joint_pos[7:] - ref_joint_pos[7:])
    
    # 插值joint_vel
    # 修复：保持x,y速度为0或目标值，避免水平漂移
    interp_joint_vel = np.zeros((num_interp_frames, ref_joint_vel.shape[0]))
    for i, alpha in enumerate(alphas):
        # Root线性速度：保持x,y速度为0（或目标值），只插值z速度
        target_lin_vel_xy = target_joint_vel[:2]  # 使用目标帧的x,y速度
        ref_lin_vel_z = ref_joint_vel[2]
        target_lin_vel_z = target_joint_vel[2]
        interp_lin_vel_z = ref_lin_vel_z + alpha * (target_lin_vel_z - ref_lin_vel_z)
        interp_joint_vel[i, :3] = np.array([target_lin_vel_xy[0], target_lin_vel_xy[1], interp_lin_vel_z])
        
        interp_joint_vel[i, 3:6] = ref_joint_vel[3:6] + alpha * (target_joint_vel[3:6] - ref_joint_vel[3:6])  # root ang vel
        interp_joint_vel[i, 6:] = ref_joint_vel[6:] + alpha * (target_joint_vel[6:] - ref_joint_vel[6:])  # joint vel
    
    # 插值body_pos_w
    # 修复：保持x,y位置为目标帧位置，只插值z
    nbody = ref_body_pos_w.shape[0]
    interp_body_pos_w = np.zeros((num_interp_frames, nbody, 3))
    for i, alpha in enumerate(alphas):
        # 使用目标帧的x,y，只插值z
        target_body_xy = target_body_pos_w[:, :2]  # 使用目标帧的x,y
        ref_body_z = ref_body_pos_w[:, 2]
        target_body_z = target_body_pos_w[:, 2]
        interp_body_z = ref_body_z + alpha * (target_body_z - ref_body_z)
        interp_body_pos_w[i, :, :2] = target_body_xy
        interp_body_pos_w[i, :, 2] = interp_body_z
    
    # 插值body_quat_w（slerp）
    interp_body_quat_w = np.zeros((num_interp_frames, nbody, 4))
    for i, alpha in enumerate(alphas):
        for j in range(nbody):
            interp_body_quat_w[i, j] = quat_slerp(ref_body_quat_w[j], target_body_quat_w[j], alpha)[0]
    
    # 插值body_lin_vel_w（线性插值）
    interp_body_lin_vel_w = np.zeros((num_interp_frames, nbody, 3))
    for i, alpha in enumerate(alphas):
        interp_body_lin_vel_w[i] = ref_body_lin_vel_w + alpha * (target_body_lin_vel_w - ref_body_lin_vel_w)
    
    # 插值body_ang_vel_w（线性插值）
    interp_body_ang_vel_w = np.zeros((num_interp_frames, nbody, 3))
    for i, alpha in enumerate(alphas):
        interp_body_ang_vel_w[i] = ref_body_ang_vel_w + alpha * (target_body_ang_vel_w - ref_body_ang_vel_w)
    
    # 合并：插值帧 + 原始输入文件
    output_data = {}
    for key in input_data.keys():
        if key in ['joint_names', 'body_names', 'fps']:
            output_data[key] = input_data[key]
        elif key == 'joint_pos':
            output_data[key] = np.concatenate([interp_joint_pos, input_data[key]], axis=0)
        elif key == 'joint_vel':
            output_data[key] = np.concatenate([interp_joint_vel, input_data[key]], axis=0)
        elif key == 'body_pos_w':
            output_data[key] = np.concatenate([interp_body_pos_w, input_data[key]], axis=0)
        elif key == 'body_quat_w':
            output_data[key] = np.concatenate([interp_body_quat_w, input_data[key]], axis=0)
        elif key == 'body_lin_vel_w':
            output_data[key] = np.concatenate([interp_body_lin_vel_w, input_data[key]], axis=0)
        elif key == 'body_ang_vel_w':
            output_data[key] = np.concatenate([interp_body_ang_vel_w, input_data[key]], axis=0)
        else:
            # 其他字段（如object相关），如果有的话
            arr = input_data[key]
            if isinstance(arr, np.ndarray) and len(arr.shape) > 0 and arr.shape[0] > 1:
                # 时间序列数据，需要插值
                ref_arr = ref_data[key]
                if ref_arr.shape[0] > reference_frame_idx:
                    ref_frame = ref_arr[reference_frame_idx]
                    target_frame = arr[0]
                    # 简单线性插值
                    interp_arr = np.zeros((num_interp_frames,) + arr.shape[1:])
                    for i, alpha in enumerate(alphas):
                        interp_arr[i] = ref_frame + alpha * (target_frame - ref_frame)
                    output_data[key] = np.concatenate([interp_arr, arr], axis=0)
                else:
                    # 如果参考文件没有这个字段，只复制输入文件
                    output_data[key] = arr
            else:
                output_data[key] = arr
    
    # 保存
    print(f"\n保存到: {output_file}")
    np.savez(output_file, **output_data)
    
    original_frames = input_data['joint_pos'].shape[0]
    new_frames = output_data['joint_pos'].shape[0]
    print(f"✓ 完成！")
    print(f"  原始帧数: {original_frames}")
    print(f"  插值帧数: {num_interp_frames}")
    print(f"  总帧数: {new_frames}")
    print(f"  总时长: {new_frames/fps:.2f}秒")

if __name__ == "__main__":
    # 从原始完整文件提取第0帧作为初始姿态
    reference_file = "converted_res/robot_only/fight1_subject2_mj_fps50.npz"
    input_file = "converted_res/robot_only/fight1_subject2_trimmed_10700_11700.npz"
    output_file = "converted_res/robot_only/fight1_subject2_trimmed_10700_11700_with_interp.npz"
    
    # 13帧插值 = 0.25秒（50fps，向上取整）
    prepend_interpolation(
        reference_file=reference_file,
        input_file=input_file,
        output_file=output_file,
        num_interp_frames=13,
        reference_frame_idx=0
    )
