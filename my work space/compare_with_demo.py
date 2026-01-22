#!/usr/bin/env python3
"""
对比修复后的数据与WBT原生demo的速度
"""
import numpy as np
from pathlib import Path

def analyze_velocity(file_path: str, label: str):
    """分析文件的速度"""
    if not Path(file_path).exists():
        print(f"⚠️  文件不存在: {file_path}")
        return None
    
    data = np.load(file_path)
    
    result = {
        "label": label,
        "file": file_path,
    }
    
    if "joint_vel" in data:
        joint_vel = data["joint_vel"]
        result["joint_vel"] = {
            "min": float(np.min(joint_vel)),
            "max": float(np.max(joint_vel)),
            "max_abs": float(np.max(np.abs(joint_vel))),
            "mean": float(np.mean(joint_vel)),
            "std": float(np.std(joint_vel)),
        }
    
    if "body_lin_vel_w" in data:
        body_lin_vel = data["body_lin_vel_w"]
        lin_vel_norm = np.linalg.norm(body_lin_vel, axis=-1)
        result["body_lin_vel"] = {
            "max": float(np.max(lin_vel_norm)),
            "mean": float(np.mean(lin_vel_norm)),
        }
    
    if "body_ang_vel_w" in data:
        body_ang_vel = data["body_ang_vel_w"]
        ang_vel_norm = np.linalg.norm(body_ang_vel, axis=-1)
        result["body_ang_vel"] = {
            "max": float(np.max(ang_vel_norm)),
            "mean": float(np.mean(ang_vel_norm)),
        }
    
    return result

def print_comparison(results):
    """打印对比结果"""
    print("=" * 100)
    print("WBT原生Demo vs 修复后的数据 - 速度对比")
    print("=" * 100)
    
    # 找到demo和修复后的数据
    demo_result = None
    fixed_result = None
    
    for r in results:
        if r:
            label_lower = r["label"].lower()
            if ("demo" in label_lower or "dancing" in label_lower or "sub3_largebox" in label_lower or "motion_crawl" in label_lower) and demo_result is None:
                demo_result = r
            elif "修复后" in r["label"] or "fixed" in label_lower:
                fixed_result = r
    
    if not demo_result:
        print("⚠️  未找到demo数据")
        return
    
    if not fixed_result:
        print("⚠️  未找到修复后的数据")
        return
    
    # 对比关节速度
    if "joint_vel" in demo_result and "joint_vel" in fixed_result:
        print(f"\n【关节速度 (joint_vel)】")
        print(f"{'指标':<20} {'Demo':<25} {'修复后':<25} {'比例':<15}")
        print("-" * 100)
        
        demo = demo_result["joint_vel"]
        fixed = fixed_result["joint_vel"]
        
        print(f"{'最大绝对值':<20} {demo['max_abs']:>10.4f} rad/s    {fixed['max_abs']:>10.4f} rad/s    {fixed['max_abs']/demo['max_abs']:>6.2f}x")
        print(f"{'最大值':<20} {demo['max']:>10.4f} rad/s    {fixed['max']:>10.4f} rad/s    {fixed['max']/demo['max']:>6.2f}x")
        print(f"{'最小值':<20} {demo['min']:>10.4f} rad/s    {fixed['min']:>10.4f} rad/s    {fixed['min']/demo['min']:>6.2f}x")
        print(f"{'标准差':<20} {demo['std']:>10.4f} rad/s    {fixed['std']:>10.4f} rad/s    {fixed['std']/demo['std']:>6.2f}x")
        
        ratio = fixed['max_abs'] / demo['max_abs']
        if ratio > 2.0:
            print(f"\n  ⚠️  警告: 修复后的速度是demo的 {ratio:.2f} 倍，仍然过大！")
        elif ratio > 1.5:
            print(f"\n  ⚠️  注意: 修复后的速度是demo的 {ratio:.2f} 倍，可能仍然偏大")
        else:
            print(f"\n  ✅ 修复后的速度与demo接近（{ratio:.2f}倍）")
    
    # 对比body速度
    if "body_lin_vel" in demo_result and "body_lin_vel" in fixed_result:
        print(f"\n【Body线性速度 (body_lin_vel_w)】")
        demo = demo_result["body_lin_vel"]
        fixed = fixed_result["body_lin_vel"]
        ratio = fixed['max'] / demo['max']
        print(f"  Demo最大: {demo['max']:.4f} m/s")
        print(f"  修复后最大: {fixed['max']:.4f} m/s")
        print(f"  比例: {ratio:.2f}x")
    
    if "body_ang_vel" in demo_result and "body_ang_vel" in fixed_result:
        print(f"\n【Body角速度 (body_ang_vel_w)】")
        demo = demo_result["body_ang_vel"]
        fixed = fixed_result["body_ang_vel"]
        ratio = fixed['max'] / demo['max']
        print(f"  Demo最大: {demo['max']:.4f} rad/s")
        print(f"  修复后最大: {fixed['max']:.4f} rad/s")
        print(f"  比例: {ratio:.2f}x")
    
    print("\n" + "=" * 100)

if __name__ == "__main__":
    # 分析所有相关文件
    files_to_analyze = [
        ("/home/wasabi/g1-motion-control/third_party/holosoma/src/holosoma/holosoma/data/motions/g1_29dof/whole_body_tracking/sub3_largebox_003_mj.npz", "Demo: sub3_largebox_003"),
        ("/home/wasabi/g1-motion-control/third_party/holosoma/src/holosoma/holosoma/data/motions/g1_29dof/whole_body_tracking/motion_crawl_slope.npz", "Demo: motion_crawl_slope"),
        ("/home/wasabi/g1-motion-control/my work space/fight1_subject3_robot_motion_wbt_fixed.npz", "修复后: fight1_subject3"),
        ("/home/wasabi/g1-motion-control/my work space/fight1_subject3_robot_motion_wbt.npz", "修复前: fight1_subject3"),
    ]
    
    results = []
    for file_path, label in files_to_analyze:
        print(f"\n分析: {label}")
        result = analyze_velocity(file_path, label)
        if result:
            results.append(result)
    
    # 打印详细对比
    print("\n" + "=" * 100)
    print("详细数据")
    print("=" * 100)
    for r in results:
        if not r:
            continue
        print(f"\n【{r['label']}】")
        if "joint_vel" in r:
            jv = r["joint_vel"]
            print(f"  关节速度: 范围=[{jv['min']:.4f}, {jv['max']:.4f}], 最大绝对值={jv['max_abs']:.4f} rad/s")
        if "body_lin_vel" in r:
            print(f"  Body线性速度: 最大={r['body_lin_vel']['max']:.4f} m/s")
        if "body_ang_vel" in r:
            print(f"  Body角速度: 最大={r['body_ang_vel']['max']:.4f} rad/s")
    
    # 对比
    print_comparison(results)
