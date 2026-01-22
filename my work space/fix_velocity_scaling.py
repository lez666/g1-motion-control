#!/usr/bin/env python3
"""
修复motion数据文件的速度缩放问题
直接对已转换的文件应用LAFAN倍率
"""
import numpy as np
from pathlib import Path

def fix_velocity_scaling(input_file: str, output_file: str, scale_factor: float = 1.27/1.7):
    """修复速度缩放"""
    print("=" * 80)
    print(f"修复速度缩放: {input_file}")
    print(f"输出文件: {output_file}")
    print(f"倍率: {scale_factor:.6f}")
    print("=" * 80)
    
    # 加载数据
    data = np.load(input_file)
    print(f"\n原始数据字段: {list(data.keys())}")
    
    # 创建新数据字典
    new_data = {}
    
    # 复制所有字段
    for key in data.keys():
        new_data[key] = data[key].copy()
    
    # 应用速度缩放
    if "joint_vel" in new_data:
        print(f"\n修复 joint_vel...")
        print(f"  原始范围: [{np.min(new_data['joint_vel']):.4f}, {np.max(new_data['joint_vel']):.4f}] rad/s")
        new_data["joint_vel"] = new_data["joint_vel"] * scale_factor
        print(f"  缩放后范围: [{np.min(new_data['joint_vel']):.4f}, {np.max(new_data['joint_vel']):.4f}] rad/s")
    
    if "body_lin_vel_w" in new_data:
        print(f"\n修复 body_lin_vel_w...")
        lin_vel_norm_orig = np.max(np.linalg.norm(new_data["body_lin_vel_w"], axis=-1))
        print(f"  原始最大速度: {lin_vel_norm_orig:.4f} m/s")
        new_data["body_lin_vel_w"] = new_data["body_lin_vel_w"] * scale_factor
        lin_vel_norm_new = np.max(np.linalg.norm(new_data["body_lin_vel_w"], axis=-1))
        print(f"  缩放后最大速度: {lin_vel_norm_new:.4f} m/s")
    
    if "body_ang_vel_w" in new_data:
        print(f"\n修复 body_ang_vel_w...")
        ang_vel_norm_orig = np.max(np.linalg.norm(new_data["body_ang_vel_w"], axis=-1))
        print(f"  原始最大角速度: {ang_vel_norm_orig:.4f} rad/s")
        new_data["body_ang_vel_w"] = new_data["body_ang_vel_w"] * scale_factor
        ang_vel_norm_new = np.max(np.linalg.norm(new_data["body_ang_vel_w"], axis=-1))
        print(f"  缩放后最大角速度: {ang_vel_norm_new:.4f} rad/s")
    
    # 保存新文件
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_file, **new_data)
    print(f"\n✅ 已保存修复后的文件: {output_file}")
    
    # 验证
    print(f"\n验证新文件...")
    verify_data = np.load(output_file)
    if "joint_vel" in verify_data:
        max_vel = np.max(np.abs(verify_data["joint_vel"]))
        print(f"  最大关节速度: {max_vel:.4f} rad/s")
        if max_vel < 60.0:
            print(f"  ✅ 速度在合理范围内")
        else:
            print(f"  ⚠️  速度仍然较大")
    
    return output_file

if __name__ == "__main__":
    # 修复fight1_subject3的文件
    input_file = "/home/wasabi/g1-motion-control/my work space/fight1_subject3_robot_motion_wbt.npz"
    output_file = "/home/wasabi/g1-motion-control/my work space/fight1_subject3_robot_motion_wbt_fixed.npz"
    
    if not Path(input_file).exists():
        print(f"错误: 输入文件不存在: {input_file}")
        exit(1)
    
    fix_velocity_scaling(input_file, output_file, scale_factor=1.27/1.7)
    
    print("\n" + "=" * 80)
    print("修复完成！")
    print("=" * 80)
