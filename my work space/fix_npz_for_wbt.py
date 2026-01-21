#!/usr/bin/env python3
"""
修复 npz 文件，添加缺失的 body_names 和 joint_names 字段
用于 holosoma whole body tracking 训练
"""

import numpy as np
from pathlib import Path

# G1 29 DOF 的关节名称（从 robot.py 配置中获取）
G1_JOINT_NAMES = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]

# G1 的 body 名称（从 isaacsim 代码注释中获取，按顺序）
# 注意：这个顺序需要与 body_pos_w 等数据的维度匹配
G1_BODY_NAMES = [
    "pelvis",
    "left_hip_yaw_link",
    "right_hip_yaw_link",
    "torso_link",
    "left_hip_roll_link",
    "right_hip_roll_link",
    "left_shoulder_pitch_link",
    "right_shoulder_pitch_link",
    "left_hip_pitch_link",
    "right_hip_pitch_link",
    "left_shoulder_roll_link",
    "right_shoulder_roll_link",
    "left_knee_link",
    "right_knee_link",
    "left_shoulder_yaw_link",
    "right_shoulder_yaw_link",
    "left_ankle_link",
    "right_ankle_link",
    "left_elbow_link",
    "right_elbow_link",
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_wrist_roll_link",
    "right_wrist_roll_link",
    "left_wrist_pitch_link",
    "right_wrist_pitch_link",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
    "left_foot_contact_point",
    "right_foot_contact_point",
    "head_link",
    "base_link",
]


def fix_npz_file(npz_path: str, output_path: str = None):
    """
    修复 npz 文件，添加缺失的字段
    
    Args:
        npz_path: 输入的 npz 文件路径
        output_path: 输出的 npz 文件路径，如果为 None 则覆盖原文件
    """
    npz_path = Path(npz_path)
    if not npz_path.exists():
        raise FileNotFoundError(f"文件不存在: {npz_path}")
    
    print(f"加载文件: {npz_path}")
    data = dict(np.load(npz_path, allow_pickle=True))
    
    print(f"当前文件包含的键: {list(data.keys())}")
    
    # 检查并添加缺失的字段
    needs_fix = False
    
    # 检查 joint_names
    if "joint_names" not in data:
        print("添加 joint_names...")
        # 检查 joint_pos 的维度
        if "joint_pos" in data:
            joint_pos = data["joint_pos"]
            # joint_pos 应该是 (T, 7+29)，前7个是 root，后面29个是关节
            if joint_pos.shape[1] >= 7 + len(G1_JOINT_NAMES):
                data["joint_names"] = np.array(G1_JOINT_NAMES, dtype=object)
                needs_fix = True
                print(f"  添加了 {len(G1_JOINT_NAMES)} 个关节名称")
            else:
                print(f"  警告: joint_pos 维度 {joint_pos.shape} 与预期不匹配")
        else:
            print("  警告: 未找到 joint_pos，无法推断 joint_names")
    else:
        print(f"joint_names 已存在: {len(data['joint_names'])} 个")
    
    # 检查 body_names
    if "body_names" not in data:
        print("添加 body_names...")
        # 检查 body_pos_w 的维度
        if "body_pos_w" in data:
            body_pos_w = data["body_pos_w"]
            n_bodies = body_pos_w.shape[1]
            if n_bodies <= len(G1_BODY_NAMES):
                # 使用前 n_bodies 个 body 名称
                data["body_names"] = np.array(G1_BODY_NAMES[:n_bodies], dtype=object)
                needs_fix = True
                print(f"  添加了 {n_bodies} 个 body 名称")
            else:
                print(f"  警告: body_pos_w 有 {n_bodies} 个 body，但配置中只有 {len(G1_BODY_NAMES)} 个")
                data["body_names"] = np.array(G1_BODY_NAMES + [f"body_{i}" for i in range(len(G1_BODY_NAMES), n_bodies)], dtype=object)
                needs_fix = True
                print(f"  添加了 {n_bodies} 个 body 名称（部分为占位符）")
        else:
            print("  警告: 未找到 body_pos_w，无法推断 body_names")
    else:
        print(f"body_names 已存在: {len(data['body_names'])} 个")
    
    # 检查 fps
    if "fps" not in data:
        print("添加 fps (默认值 60)...")
        data["fps"] = np.array([60.0])
        needs_fix = True
    else:
        print(f"fps 已存在: {data['fps']}")
    
    if not needs_fix:
        print("文件已经包含所有必需的字段，无需修复")
        return
    
    # 保存修复后的文件
    if output_path is None:
        output_path = npz_path
    else:
        output_path = Path(output_path)
    
    print(f"\n保存修复后的文件到: {output_path}")
    np.savez(output_path, **data)
    print("完成！")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("用法: python fix_npz_for_wbt.py <npz_file> [output_file]")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    fix_npz_file(input_file, output_file)
