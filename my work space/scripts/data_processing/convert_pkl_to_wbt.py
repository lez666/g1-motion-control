#!/usr/bin/env python3
"""
将.pkl文件转换为WBT训练格式的.npz文件，并自动应用速度缩放以匹配demo标准。

用法:
    python convert_pkl_to_wbt.py --pkl_path input.pkl --output_path output.npz --fps 50
"""
import sys
from pathlib import Path

# 添加路径
script_dir = Path(__file__).parent
project_root = script_dir.parent.parent.parent  # 从scripts/data_processing到项目根目录
holosoma_root = project_root / "third_party" / "holosoma"
sys.path.insert(0, str(holosoma_root / "src" / "holosoma_retargeting"))

from convert_pkl_to_npz import convert_pkl_to_npz
import numpy as np

def get_demo_velocity_standard(demo_file=None):
    """获取Demo的速度标准"""
    if demo_file is None:
        # 使用相对路径
        project_root = Path(__file__).parent.parent.parent.parent
        demo_file = project_root / "third_party/holosoma/src/holosoma/holosoma/data/motions/g1_29dof/whole_body_tracking/sub3_largebox_003_mj.npz"
    demo_file = str(demo_file)
    data = np.load(demo_file)
    joint_vel = data["joint_vel"]
    dof_vel = joint_vel[:, 6:]  # 只取DOF速度
    max_dof_vel = np.max(np.abs(dof_vel))
    
    body_lin_vel = data["body_lin_vel_w"]
    max_lin_vel = np.max(np.linalg.norm(body_lin_vel, axis=-1))
    
    body_ang_vel = data["body_ang_vel_w"]
    max_ang_vel = np.max(np.linalg.norm(body_ang_vel, axis=-1))
    
    return {
        "max_joint_vel": max_dof_vel,
        "max_body_lin_vel": max_lin_vel,
        "max_body_ang_vel": max_ang_vel,
    }

def calculate_velocity_scale(temp_file, demo_standard):
    """计算速度缩放因子
    
    策略：优先匹配DOF速度（关节速度），这是最重要的。
    Body速度可能因为计算方式不同而有差异，所以只作为参考。
    """
    data = np.load(temp_file)
    joint_vel = data["joint_vel"]
    dof_vel = joint_vel[:, 6:]
    max_dof_vel = np.max(np.abs(dof_vel))
    
    body_lin_vel = data["body_lin_vel_w"]
    # 排除root body（索引0），因为root速度已经在joint_vel中
    # 只检查其他body的速度
    if body_lin_vel.shape[1] > 1:
        body_lin_vel_other = body_lin_vel[:, 1:]  # 排除root
        max_lin_vel = np.max(np.linalg.norm(body_lin_vel_other, axis=-1))
    else:
        max_lin_vel = np.max(np.linalg.norm(body_lin_vel, axis=-1))
    
    body_ang_vel = data["body_ang_vel_w"]
    if body_ang_vel.shape[1] > 1:
        body_ang_vel_other = body_ang_vel[:, 1:]  # 排除root
        max_ang_vel = np.max(np.linalg.norm(body_ang_vel_other, axis=-1))
    else:
        max_ang_vel = np.max(np.linalg.norm(body_ang_vel, axis=-1))
    
    # 计算各个缩放因子
    scale_joint = demo_standard["max_joint_vel"] / max_dof_vel
    scale_lin = demo_standard["max_body_lin_vel"] / max_lin_vel if max_lin_vel > 0 else 1.0
    scale_ang = demo_standard["max_body_ang_vel"] / max_ang_vel if max_ang_vel > 0 else 1.0
    
    # 优先使用DOF速度的缩放因子（这是最重要的）
    # 但如果Body速度缩放因子更小，说明Body速度异常大，需要检查
    scale = scale_joint
    if scale_lin < scale_joint * 0.5:  # Body速度异常大
        print(f"⚠️  警告: Body线性速度异常大 ({max_lin_vel:.4f} m/s)，可能计算有误")
        print(f"   将使用DOF速度缩放因子: {scale_joint:.6f}")
    elif scale_lin < scale_joint:
        print(f"⚠️  注意: Body线性速度较大，但将优先匹配DOF速度")
    
    print(f"\n速度分析:")
    print(f"  当前最大DOF速度: {max_dof_vel:.4f} rad/s")
    print(f"  当前最大Body线性速度（排除root）: {max_lin_vel:.4f} m/s")
    print(f"  当前最大Body角速度（排除root）: {max_ang_vel:.4f} rad/s")
    print(f"\nDemo标准:")
    print(f"  最大DOF速度: {demo_standard['max_joint_vel']:.4f} rad/s")
    print(f"  最大Body线性速度: {demo_standard['max_body_lin_vel']:.4f} m/s")
    print(f"  最大Body角速度: {demo_standard['max_body_ang_vel']:.4f} rad/s")
    print(f"\n计算的缩放因子:")
    print(f"  基于DOF速度: {scale_joint:.6f} ⭐ (将使用此值)")
    print(f"  基于Body线性速度: {scale_lin:.6f}")
    print(f"  基于Body角速度: {scale_ang:.6f}")
    print(f"  最终使用: {scale:.6f}")
    
    return scale

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="将.pkl文件转换为WBT训练格式的.npz文件")
    parser.add_argument("--pkl_path", type=str, required=True, help="输入的.pkl文件路径")
    parser.add_argument("--output_path", type=str, required=True, help="输出的.npz文件路径")
    parser.add_argument("--fps", type=float, default=50, help="输出FPS（默认50）")
    parser.add_argument("--reference_demo", type=str, default=None, help="参考demo文件路径（可选）")
    parser.add_argument("--xml_path", type=str, default="models/g1/g1_29dof.xml", help="机器人XML文件路径")
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("转换 .pkl 为 WBT 格式（按Demo标准）")
    print("=" * 80)
    
    # 1. 获取Demo标准
    print("\n[步骤1] 获取Demo速度标准...")
    demo_standard = get_demo_velocity_standard(args.reference_demo)
    
    # 2. 先转换一次（不缩放）获取速度范围
    print("\n[步骤2] 转换pkl文件（临时，用于计算缩放因子）...")
    import tempfile
    temp_file = tempfile.NamedTemporaryFile(suffix='.npz', delete=False).name
    convert_pkl_to_npz(
        args.pkl_path,
        temp_file,
        args.xml_path,
        fps=args.fps,
        smooth_velocities=True,
        smooth_window=5,
        velocity_scale=None,
    )
    
    # 3. 计算缩放因子
    print("\n[步骤3] 计算速度缩放因子...")
    scale = calculate_velocity_scale(temp_file, demo_standard)
    
    # 4. 使用正确的缩放因子重新转换
    print(f"\n[步骤4] 使用缩放因子 {scale:.6f} 重新转换...")
    convert_pkl_to_npz(
        args.pkl_path,
        args.output_path,
        args.xml_path,
        fps=args.fps,
        smooth_velocities=True,
        smooth_window=5,
        velocity_scale=scale,
    )
    
    # 清理临时文件
    import os
    os.unlink(temp_file)
    
    # 5. 检查Body线性速度，如果还是太大，需要额外缩放
    print("\n[步骤5] 检查Body线性速度...")
    final_data = np.load(output_file)
    body_lin_vel = final_data["body_lin_vel_w"]
    max_lin_vel = np.max(np.linalg.norm(body_lin_vel, axis=-1))
    
    # 如果Body线性速度还是太大（>4 m/s），需要额外缩放
    if max_lin_vel > 4.0:
        print(f"⚠️  Body线性速度仍然太大: {max_lin_vel:.4f} m/s (目标: <4.0 m/s)")
        print(f"   需要额外缩放Body速度...")
        
        # 计算Body速度的额外缩放因子（目标：3.5 m/s，留一些余量）
        body_scale = 3.5 / max_lin_vel
        print(f"   Body速度额外缩放因子: {body_scale:.6f}")
        
        # 重新加载数据并应用Body速度缩放
        final_data_dict = {}
        for k, v in final_data.items():
            if k in ['body_lin_vel_w', 'body_ang_vel_w']:
                final_data_dict[k] = (v * body_scale).astype(np.float32)
            elif k in ['joint_pos', 'joint_vel', 'body_pos_w', 'body_quat_w']:
                final_data_dict[k] = v.astype(np.float32)
            else:
                final_data_dict[k] = v  # joint_names, body_names, fps等保持原样
        
        # 保存修正后的数据
        np.savez(output_file, **final_data_dict)
        print(f"   ✅ 已应用Body速度缩放，重新保存文件")
    
    # 6. 验证最终结果
    print("\n[步骤6] 验证最终结果...")
    final_data = np.load(output_file)
    joint_vel = final_data["joint_vel"]
    dof_vel = joint_vel[:, 6:]
    max_dof_vel = np.max(np.abs(dof_vel))
    
    body_lin_vel = final_data["body_lin_vel_w"]
    max_lin_vel = np.max(np.linalg.norm(body_lin_vel, axis=-1))
    
    body_ang_vel = final_data["body_ang_vel_w"]
    max_ang_vel = np.max(np.linalg.norm(body_ang_vel, axis=-1))
    
    print(f"\n最终速度:")
    print(f"  最大DOF速度: {max_dof_vel:.4f} rad/s (Demo: {demo_standard['max_joint_vel']:.4f})")
    print(f"  最大Body线性速度: {max_lin_vel:.4f} m/s (Demo: {demo_standard['max_body_lin_vel']:.4f}, 目标: <4.0)")
    print(f"  最大Body角速度: {max_ang_vel:.4f} rad/s (Demo: {demo_standard['max_body_ang_vel']:.4f})")
    
    # 检查是否匹配
    if abs(max_dof_vel - demo_standard['max_joint_vel']) < 1.0:
        print(f"\n✅ DOF速度与Demo匹配！")
    else:
        print(f"\n⚠️  DOF速度与Demo有差异: {abs(max_dof_vel - demo_standard['max_joint_vel']):.4f} rad/s")
    
    if max_lin_vel < 4.0:
        print(f"✅ Body线性速度在可接受范围内 (<4.0 m/s)")
    else:
        print(f"⚠️  Body线性速度仍然较大: {max_lin_vel:.4f} m/s")
    
    print(f"\n✅ 转换完成！文件保存至: {args.output_path}")
