#!/usr/bin/env python3
"""将 npz 文件转换为 holosoma whole body tracking 格式"""

import numpy as np
import sys

G1_JOINT_NAMES = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]

# 从 IsaacSim 加载的实际 body 名称列表（32 个）
G1_BODY_NAMES = [
    'pelvis', 'left_hip_yaw_link', 'right_hip_yaw_link', 'torso_link',
    'left_hip_roll_link', 'right_hip_roll_link', 'left_shoulder_pitch_link', 'right_shoulder_pitch_link',
    'left_hip_pitch_link', 'right_hip_pitch_link', 'left_shoulder_roll_link', 'right_shoulder_roll_link',
    'left_knee_link', 'right_knee_link', 'left_shoulder_yaw_link', 'right_shoulder_yaw_link',
    'left_ankle_pitch_link', 'right_ankle_pitch_link', 'left_elbow_link', 'right_elbow_link',
    'left_ankle_roll_link', 'right_ankle_roll_link', 'left_wrist_roll_link', 'right_wrist_roll_link',
    'left_wrist_pitch_link', 'right_wrist_pitch_link', 'left_wrist_yaw_link', 'right_wrist_yaw_link',
    'left_foot_contact_point', 'right_foot_contact_point', 'head_link', 'base_link'
]

def convert_to_wbt_format(input_file, output_file):
    print(f"加载原始文件: {input_file}")
    data = np.load(input_file, allow_pickle=True)
    
    print("原始文件包含的键:")
    for k in data.keys():
        val = data[k]
        if hasattr(val, 'shape'):
            print(f"  {k}: shape={val.shape}, dtype={val.dtype}")
    
    joint_pos = data['joint_pos']
    joint_vel = data['joint_vel']
    n_frames = joint_pos.shape[0]
    n_bodies = len(G1_BODY_NAMES)
    
    print(f"\n帧数: {n_frames}, 需要的 body 数量: {n_bodies}")
    
    if 'body_pos_w' in data:
        body_pos_w_orig = data['body_pos_w']
        body_quat_w_orig = data['body_quat_w']
        body_lin_vel_w_orig = data['body_lin_vel_w']
        body_ang_vel_w_orig = data['body_ang_vel_w']
        n_bodies_in_file = body_pos_w_orig.shape[1]
        
        ang_vel_dim = body_ang_vel_w_orig.shape[2] if len(body_ang_vel_w_orig.shape) > 2 else 3
        print(f"文件中的 body 数量: {n_bodies_in_file}, ang_vel 维度: {ang_vel_dim}")
        
        body_pos_w = np.zeros((n_frames, n_bodies, 3), dtype=np.float32)
        body_quat_w = np.zeros((n_frames, n_bodies, 4), dtype=np.float32)
        body_lin_vel_w = np.zeros((n_frames, n_bodies, 3), dtype=np.float32)
        body_ang_vel_w = np.zeros((n_frames, n_bodies, 3), dtype=np.float32)
        
        min_bodies = min(n_bodies_in_file, n_bodies)
        body_pos_w[:, :min_bodies, :] = body_pos_w_orig[:, :min_bodies, :]
        body_quat_w[:, :min_bodies, :] = body_quat_w_orig[:, :min_bodies, :]
        body_lin_vel_w[:, :min_bodies, :] = body_lin_vel_w_orig[:, :min_bodies, :]
        
        if ang_vel_dim == 4:
            body_ang_vel_w[:, :min_bodies, :] = body_ang_vel_w_orig[:, :min_bodies, :3]
        else:
            body_ang_vel_w[:, :min_bodies, :] = body_ang_vel_w_orig[:, :min_bodies, :]
        
        for i in range(min_bodies, n_bodies):
            body_pos_w[:, i, :] = body_pos_w[:, 0, :]
            body_quat_w[:, i, :] = body_quat_w[:, 0, :]
            body_lin_vel_w[:, i, :] = body_lin_vel_w[:, 0, :]
            body_ang_vel_w[:, i, :] = body_ang_vel_w[:, 0, :]
        
        for i in range(n_bodies):
            if np.allclose(body_quat_w[:, i, :], 0):
                body_quat_w[:, i, 0] = 1.0
    else:
        print("文件缺少 body 数据，从 joint 数据生成")
        body_pos_w = np.zeros((n_frames, n_bodies, 3), dtype=np.float32)
        body_quat_w = np.zeros((n_frames, n_bodies, 4), dtype=np.float32)
        body_lin_vel_w = np.zeros((n_frames, n_bodies, 3), dtype=np.float32)
        body_ang_vel_w = np.zeros((n_frames, n_bodies, 3), dtype=np.float32)
        for i in range(n_bodies):
            body_pos_w[:, i, :] = joint_pos[:, :3]
            body_quat_w[:, i, 0] = 1.0
            body_lin_vel_w[:, i, :] = joint_vel[:, :3]
            body_ang_vel_w[:, i, :] = joint_vel[:, 3:6]
    
    fps = data['fps'][0] if 'fps' in data else 60
    print(f"FPS: {fps}")
    
    max_jn = max(len(n) for n in G1_JOINT_NAMES)
    max_bn = max(len(n) for n in G1_BODY_NAMES)
    joint_names_array = np.array(G1_JOINT_NAMES, dtype=f'U{max_jn + 1}')
    body_names_array = np.array(G1_BODY_NAMES, dtype=f'U{max_bn + 1}')
    
    output_data = {
        'fps': np.array([fps]),
        'joint_pos': joint_pos.astype(np.float32),
        'joint_vel': joint_vel.astype(np.float32),
        'body_pos_w': body_pos_w,
        'body_quat_w': body_quat_w,
        'body_lin_vel_w': body_lin_vel_w,
        'body_ang_vel_w': body_ang_vel_w,
        'joint_names': joint_names_array,
        'body_names': body_names_array,
    }
    
    print(f"\n保存: {output_file}")
    np.savez(output_file, **output_data)
    
    print("\n转换后的数据:")
    for k, v in output_data.items():
        if hasattr(v, 'shape'):
            print(f"  {k}: shape={v.shape}, dtype={v.dtype}")
    print("\n转换完成!")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("用法: python convert_npz_to_wbt_format.py <input.npz> <output.npz>")
        sys.exit(1)
    convert_to_wbt_format(sys.argv[1], sys.argv[2])
