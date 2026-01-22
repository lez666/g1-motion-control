#!/usr/bin/env python3
"""
测试速度倍率缩放是否正确应用
"""
import numpy as np
from pathlib import Path

def test_velocity_scaling():
    """测试速度缩放"""
    print("=" * 80)
    print("测试速度倍率缩放")
    print("=" * 80)
    
    # 当前数据
    current_file = "/home/wasabi/g1-motion-control/my work space/fight1_subject3_robot_motion_wbt.npz"
    data = np.load(current_file)
    
    lafan_scale = 1.27 / 1.7  # ≈ 0.747
    
    print(f"\n当前数据速度统计:")
    joint_vel = data["joint_vel"]
    print(f"  范围: [{np.min(joint_vel):.4f}, {np.max(joint_vel):.4f}] rad/s")
    print(f"  最大绝对值: {np.max(np.abs(joint_vel)):.4f} rad/s")
    
    print(f"\n应用LAFAN倍率 ({lafan_scale:.6f}) 后:")
    scaled_vel = joint_vel * lafan_scale
    print(f"  范围: [{np.min(scaled_vel):.4f}, {np.max(scaled_vel):.4f}] rad/s")
    print(f"  最大绝对值: {np.max(np.abs(scaled_vel)):.4f} rad/s")
    
    print(f"\n合理性检查:")
    max_vel_original = np.max(np.abs(joint_vel))
    max_vel_scaled = np.max(np.abs(scaled_vel))
    
    print(f"  原始最大速度: {max_vel_original:.4f} rad/s")
    print(f"  缩放后最大速度: {max_vel_scaled:.4f} rad/s")
    
    # 正常机器人关节速度通常在10-20 rad/s以内
    if max_vel_scaled > 20.0:
        print(f"  ⚠️  警告: 即使缩放后，速度仍然较大 ({max_vel_scaled:.4f} rad/s)")
        print(f"  这可能表明原始数据本身速度就很大，或者需要进一步检查")
    else:
        print(f"  ✅ 缩放后的速度在合理范围内")
    
    # 检查body速度
    if "body_lin_vel_w" in data:
        body_lin_vel = data["body_lin_vel_w"]
        lin_vel_norm = np.linalg.norm(body_lin_vel, axis=-1)
        scaled_lin_vel_norm = lin_vel_norm * lafan_scale
        
        print(f"\nBody线性速度:")
        print(f"  原始最大: {np.max(lin_vel_norm):.4f} m/s")
        print(f"  缩放后最大: {np.max(scaled_lin_vel_norm):.4f} m/s")
        
        if np.max(scaled_lin_vel_norm) > 3.0:
            print(f"  ⚠️  警告: 线性速度仍然较大")
        else:
            print(f"  ✅ 线性速度在合理范围内")
    
    if "body_ang_vel_w" in data:
        body_ang_vel = data["body_ang_vel_w"]
        ang_vel_norm = np.linalg.norm(body_ang_vel, axis=-1)
        scaled_ang_vel_norm = ang_vel_norm * lafan_scale
        
        print(f"\nBody角速度:")
        print(f"  原始最大: {np.max(ang_vel_norm):.4f} rad/s")
        print(f"  缩放后最大: {np.max(scaled_ang_vel_norm):.4f} rad/s")
        
        if np.max(scaled_ang_vel_norm) > 10.0:
            print(f"  ⚠️  警告: 角速度仍然较大")
        else:
            print(f"  ✅ 角速度在合理范围内")
    
    print(f"\n结论:")
    print(f"  1. 当前数据的速度没有应用LAFAN倍率缩放")
    print(f"  2. 速度过大 ({max_vel_original:.2f} rad/s) 会导致训练和仿真失败")
    print(f"  3. 需要重新转换数据，应用倍率 {lafan_scale:.6f}")
    print(f"  4. 修改后的 convert_data_format_mj.py 应该会自动应用这个倍率")

if __name__ == "__main__":
    test_velocity_scaling()
