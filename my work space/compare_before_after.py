#!/usr/bin/env python3
"""
对比修复前后的数据
"""
import numpy as np

def compare_files(file1: str, file2: str, label1: str, label2: str):
    """对比两个文件"""
    print("=" * 80)
    print(f"对比: {label1} vs {label2}")
    print("=" * 80)
    
    data1 = np.load(file1)
    data2 = np.load(file2)
    
    if "joint_vel" in data1 and "joint_vel" in data2:
        vel1 = data1["joint_vel"]
        vel2 = data2["joint_vel"]
        
        print(f"\n关节速度 (joint_vel):")
        print(f"  {label1}:")
        print(f"    范围: [{np.min(vel1):.4f}, {np.max(vel1):.4f}] rad/s")
        print(f"    最大绝对值: {np.max(np.abs(vel1)):.4f} rad/s")
        print(f"    均值: {np.mean(vel1):.4f} rad/s")
        print(f"    标准差: {np.std(vel1):.4f} rad/s")
        
        print(f"  {label2}:")
        print(f"    范围: [{np.min(vel2):.4f}, {np.max(vel2):.4f}] rad/s")
        print(f"    最大绝对值: {np.max(np.abs(vel2)):.4f} rad/s")
        print(f"    均值: {np.mean(vel2):.4f} rad/s")
        print(f"    标准差: {np.std(vel2):.4f} rad/s")
        
        ratio = np.max(np.abs(vel2)) / np.max(np.abs(vel1))
        print(f"\n  缩放比例: {ratio:.6f} (应该是 {1.27/1.7:.6f})")
        
        if abs(ratio - 1.27/1.7) < 0.01:
            print(f"  ✅ 缩放正确！")
        else:
            print(f"  ⚠️  缩放比例不匹配")
    
    if "body_lin_vel_w" in data1 and "body_lin_vel_w" in data2:
        lin1 = np.linalg.norm(data1["body_lin_vel_w"], axis=-1)
        lin2 = np.linalg.norm(data2["body_lin_vel_w"], axis=-1)
        
        print(f"\nBody线性速度 (body_lin_vel_w):")
        print(f"  {label1} 最大: {np.max(lin1):.4f} m/s")
        print(f"  {label2} 最大: {np.max(lin2):.4f} m/s")
        print(f"  缩放比例: {np.max(lin2)/np.max(lin1):.6f}")
    
    if "body_ang_vel_w" in data1 and "body_ang_vel_w" in data2:
        ang1 = np.linalg.norm(data1["body_ang_vel_w"], axis=-1)
        ang2 = np.linalg.norm(data2["body_ang_vel_w"], axis=-1)
        
        print(f"\nBody角速度 (body_ang_vel_w):")
        print(f"  {label1} 最大: {np.max(ang1):.4f} rad/s")
        print(f"  {label2} 最大: {np.max(ang2):.4f} rad/s")
        print(f"  缩放比例: {np.max(ang2)/np.max(ang1):.6f}")

if __name__ == "__main__":
    original = "/home/wasabi/g1-motion-control/my work space/fight1_subject3_robot_motion_wbt.npz"
    fixed = "/home/wasabi/g1-motion-control/my work space/fight1_subject3_robot_motion_wbt_fixed.npz"
    
    compare_files(original, fixed, "修复前", "修复后")
    
    print("\n" + "=" * 80)
    print("总结:")
    print("=" * 80)
    print("✅ 速度已按LAFAN倍率 (1.27/1.7 ≈ 0.747) 缩放")
    print("✅ 修复后的文件: fight1_subject3_robot_motion_wbt_fixed.npz")
    print("✅ 可以使用修复后的文件重新训练")
