#!/usr/bin/env python3
"""
创建包含默认初始姿态的单帧npz文件，用于可视化
"""
import numpy as np

# 默认关节角度（从配置中）
default_joint_angles = {
    "left_hip_pitch_joint": -0.312,
    "left_hip_roll_joint": 0.0,
    "left_hip_yaw_joint": 0.0,
    "left_knee_joint": 0.669,
    "left_ankle_pitch_joint": -0.363,
    "left_ankle_roll_joint": 0.0,
    "right_hip_pitch_joint": -0.312,
    "right_hip_roll_joint": 0.0,
    "right_hip_yaw_joint": 0.0,
    "right_knee_joint": 0.669,
    "right_ankle_pitch_joint": -0.363,
    "right_ankle_roll_joint": 0.0,
    "waist_yaw_joint": 0.0,
    "waist_roll_joint": 0.0,
    "waist_pitch_joint": 0.0,
    "left_shoulder_pitch_joint": 0.2,
    "left_shoulder_roll_joint": 0.2,
    "left_shoulder_yaw_joint": 0.0,
    "left_elbow_joint": 0.6,
    "left_wrist_roll_joint": 0.0,
    "left_wrist_pitch_joint": 0.0,
    "left_wrist_yaw_joint": 0.0,
    "right_shoulder_pitch_joint": 0.2,
    "right_shoulder_roll_joint": -0.2,
    "right_shoulder_yaw_joint": 0.0,
    "right_elbow_joint": 0.6,
    "right_wrist_roll_joint": 0.0,
    "right_wrist_pitch_joint": 0.0,
    "right_wrist_yaw_joint": 0.0,
}

# 默认root状态
default_root_pos = [0.0, 0.0, 0.76]  # x, y, z
default_root_quat = [0.0, 0.0, 0.0, 1.0]  # w, x, y, z (wxyz格式)

def create_default_pose_npz(reference_file, output_file):
    """
    从参考文件中获取关节名称和body名称，创建包含默认姿态的单帧npz文件
    """
    print(f"加载参考文件获取结构: {reference_file}")
    ref_data = np.load(reference_file, allow_pickle=True)
    
    joint_names = ref_data["joint_names"].tolist()
    body_names = ref_data["body_names"].tolist()
    nbody = len(body_names)
    ndof = len(joint_names)
    
    print(f"关节数量: {ndof}")
    print(f"Body数量: {nbody}")
    
    # 创建关节角度数组（按照joint_names的顺序）
    joint_angles = np.zeros(ndof)
    for i, joint_name in enumerate(joint_names):
        if joint_name in default_joint_angles:
            joint_angles[i] = default_joint_angles[joint_name]
        else:
            print(f"警告: 未找到关节 {joint_name} 的默认角度，使用0.0")
    
    # 创建joint_pos: [root_xyz(3), root_quat_wxyz(4), joint_angles(ndof)]
    joint_pos = np.concatenate([
        default_root_pos,  # [3]
        default_root_quat,  # [4] wxyz
        joint_angles       # [ndof]
    ]).reshape(1, -1)  # (1, 7+ndof)
    
    # 创建joint_vel: [root_lin_vel(3), root_ang_vel(3), joint_vel(ndof)]
    joint_vel = np.zeros((1, 6 + ndof))
    
    # 创建body状态（需要forward kinematics，这里先用零值，可视化工具会计算）
    # 注意：实际的body_pos_w和body_quat_w需要通过FK计算，这里创建占位符
    body_pos_w = np.zeros((1, nbody, 3))
    body_quat_w = np.zeros((1, nbody, 4))  # wxyz格式
    body_lin_vel_w = np.zeros((1, nbody, 3))
    body_ang_vel_w = np.zeros((1, nbody, 3))
    
    # 设置pelvis（第一个body）的位置和姿态
    body_pos_w[0, 0] = default_root_pos
    body_quat_w[0, 0] = default_root_quat  # wxyz
    
    # 保存为npz文件
    print(f"\n创建默认姿态文件: {output_file}")
    np.savez(
        output_file,
        joint_pos=joint_pos,
        joint_vel=joint_vel,
        body_pos_w=body_pos_w,
        body_quat_w=body_quat_w,
        body_lin_vel_w=body_lin_vel_w,
        body_ang_vel_w=body_ang_vel_w,
        joint_names=np.array(joint_names),
        body_names=np.array(body_names),
        fps=np.array([50.0])
    )
    
    print("✓ 完成！")
    print(f"\n关节角度:")
    for i, joint_name in enumerate(joint_names):
        print(f"  {joint_name}: {joint_angles[i]:.6f}")

if __name__ == "__main__":
    # 使用现有的motion文件作为参考，获取关节和body名称
    reference_file = "converted_res/robot_only/fight1_subject2_mj_fps50.npz"
    output_file = "converted_res/robot_only/default_initial_pose.npz"
    
    create_default_pose_npz(reference_file, output_file)
