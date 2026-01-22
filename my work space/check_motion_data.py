#!/usr/bin/env python3
"""
检查motion数据文件，分析速度和位置的关系
"""
import numpy as np
from pathlib import Path

def analyze_motion_file(motion_file: str):
    """分析motion文件"""
    print("=" * 80)
    print(f"分析文件: {motion_file}")
    print("=" * 80)
    
    data = np.load(motion_file)
    
    # 检查所有字段
    print(f"\n数据字段: {list(data.keys())}")
    
    # 分析位置数据
    if "joint_pos" in data:
        joint_pos = data["joint_pos"]
        print(f"\n[joint_pos] shape: {joint_pos.shape}")
        print(f"  范围: [{np.min(joint_pos):.4f}, {np.max(joint_pos):.4f}]")
        
        # 计算位置变化率（粗略的速度估计）
        if joint_pos.shape[0] > 1:
            pos_diff = np.diff(joint_pos, axis=0)
            if "fps" in data:
                fps = float(data["fps"].item() if hasattr(data["fps"], 'item') else data["fps"])
                dt = 1.0 / fps
                estimated_vel = pos_diff / dt
                print(f"  从位置差分估计的速度范围: [{np.min(estimated_vel):.4f}, {np.max(estimated_vel):.4f}] rad/s")
    
    # 分析速度数据
    if "joint_vel" in data:
        joint_vel = data["joint_vel"]
        print(f"\n[joint_vel] shape: {joint_vel.shape}")
        print(f"  范围: [{np.min(joint_vel):.4f}, {np.max(joint_vel):.4f}] rad/s")
        print(f"  均值: {np.mean(joint_vel):.4f} rad/s")
        print(f"  标准差: {np.std(joint_vel):.4f} rad/s")
        print(f"  最大绝对值: {np.max(np.abs(joint_vel)):.4f} rad/s")
        
        # 检查是否有异常大的速度
        max_vel = np.max(np.abs(joint_vel))
        if max_vel > 20.0:
            print(f"  ⚠️  警告: 速度过大！最大速度 {max_vel:.4f} rad/s")
    
    # 分析body速度
    if "body_lin_vel_w" in data:
        body_lin_vel = data["body_lin_vel_w"]
        print(f"\n[body_lin_vel_w] shape: {body_lin_vel.shape}")
        lin_vel_norm = np.linalg.norm(body_lin_vel, axis=-1)
        print(f"  速度大小范围: [{np.min(lin_vel_norm):.4f}, {np.max(lin_vel_norm):.4f}] m/s")
        print(f"  最大速度: {np.max(lin_vel_norm):.4f} m/s")
        
        if np.max(lin_vel_norm) > 5.0:
            print(f"  ⚠️  警告: 线性速度过大！")
    
    if "body_ang_vel_w" in data:
        body_ang_vel = data["body_ang_vel_w"]
        print(f"\n[body_ang_vel_w] shape: {body_ang_vel.shape}")
        ang_vel_norm = np.linalg.norm(body_ang_vel, axis=-1)
        print(f"  角速度大小范围: [{np.min(ang_vel_norm):.4f}, {np.max(ang_vel_norm):.4f}] rad/s")
        print(f"  最大角速度: {np.max(ang_vel_norm):.4f} rad/s")
        
        if np.max(ang_vel_norm) > 10.0:
            print(f"  ⚠️  警告: 角速度过大！")
    
    # 检查FPS
    if "fps" in data:
        fps = float(data["fps"].item() if hasattr(data["fps"], 'item') else data["fps"])
        print(f"\n[FPS] {fps}")
        dt = 1.0 / fps
        print(f"  时间步长: {dt:.4f} s")
    
    # 检查速度和位置的一致性
    if "joint_pos" in data and "joint_vel" in data and "fps" in data:
        print(f"\n[一致性检查]")
        joint_pos = data["joint_pos"]
        joint_vel = data["joint_vel"]
        fps = float(data["fps"].item() if hasattr(data["fps"], 'item') else data["fps"])
        dt = 1.0 / fps
        
        # 从位置计算速度
        if joint_pos.shape[0] > 1:
            pos_derivative = np.gradient(joint_pos, dt, axis=0)
            
            # 比较存储的速度和计算的速度
            if joint_vel.shape == pos_derivative.shape:
                vel_diff = np.abs(joint_vel - pos_derivative)
                max_diff = np.max(vel_diff)
                mean_diff = np.mean(vel_diff)
                
                print(f"  存储速度 vs 计算速度的最大差异: {max_diff:.6f} rad/s")
                print(f"  平均差异: {mean_diff:.6f} rad/s")
                
                if max_diff > 0.1:
                    print(f"  ⚠️  警告: 速度和位置不一致！差异过大")
                    print(f"  这可能表明速度没有正确缩放")
                else:
                    print(f"  ✅ 速度和位置基本一致")
    
    # 检查LAFAN倍率
    print(f"\n[倍率检查]")
    print(f"  LAFAN默认倍率: 1.27/1.7 ≈ {1.27/1.7:.6f}")
    print(f"  如果速度应该按此倍率缩放，但实际没有缩放，")
    print(f"  那么速度会过大 {1.0/(1.27/1.7):.2f} 倍")

def compare_files(file1: str, file2: str):
    """比较两个motion文件"""
    print("\n" + "=" * 80)
    print("比较两个motion文件")
    print("=" * 80)
    
    data1 = np.load(file1)
    data2 = np.load(file2)
    
    if "joint_vel" in data1 and "joint_vel" in data2:
        vel1 = data1["joint_vel"]
        vel2 = data2["joint_vel"]
        
        print(f"\n文件1速度范围: [{np.min(vel1):.4f}, {np.max(vel1):.4f}]")
        print(f"文件2速度范围: [{np.min(vel2):.4f}, {np.max(vel2):.4f}]")
        
        if vel1.shape == vel2.shape:
            ratio = vel2 / (vel1 + 1e-8)
            print(f"速度比例 (文件2/文件1): 均值={np.mean(ratio):.4f}, 范围=[{np.min(ratio):.4f}, {np.max(ratio):.4f}]")

if __name__ == "__main__":
    # 分析主要的motion文件
    motion_file = "/home/wasabi/g1-motion-control/my work space/fight1_subject3_robot_motion_wbt.npz"
    
    if Path(motion_file).exists():
        analyze_motion_file(motion_file)
    else:
        print(f"文件不存在: {motion_file}")
    
    # 如果有其他版本，也分析
    for i in [2, 3]:
        other_file = f"/home/wasabi/g1-motion-control/my work space/fight1_subject3_robot_motion_wbt{i}.npz"
        if Path(other_file).exists():
            print("\n" + "-" * 80)
            analyze_motion_file(other_file)
            if Path(motion_file).exists():
                compare_files(motion_file, other_file)
