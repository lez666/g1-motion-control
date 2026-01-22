#!/usr/bin/env python3
"""
修复motion数据，使其符合demo标准：
1. 将Root位置平移到接近原点（参考demo的初始位置）
2. 同时平移所有body位置
3. 检查并报告所有参数
"""
import numpy as np
import sys
from pathlib import Path

def fix_motion_to_demo_standard(input_file, output_file, reference_demo_file=None):
    """
    修复motion数据使其符合demo标准
    
    Args:
        input_file: 输入的npz文件路径
        output_file: 输出的npz文件路径
        reference_demo_file: 参考demo文件（用于获取初始位置参考）
    """
    print("=" * 80)
    print("修复motion数据到Demo标准")
    print("=" * 80)
    
    # 加载输入文件
    print(f"\n[步骤1] 加载输入文件: {input_file}")
    data = np.load(input_file, allow_pickle=True)
    
    # 加载参考demo文件
    if reference_demo_file is None:
        # 使用相对路径
        project_root = Path(__file__).parent.parent.parent.parent
        reference_demo_file = project_root / "third_party/holosoma/src/holosoma/holosoma/data/motions/g1_29dof/whole_body_tracking/sub3_largebox_003_mj.npz"
    reference_demo_file = str(reference_demo_file)
    
    print(f"[步骤2] 加载参考demo文件: {reference_demo_file}")
    demo_data = np.load(reference_demo_file, allow_pickle=True)
    demo_root_pos = demo_data['joint_pos'][:, 0:3]
    demo_initial_pos = demo_root_pos[0]
    
    print(f"  Demo初始位置: {demo_initial_pos}")
    
    # 获取当前数据的初始位置
    current_root_pos = data['joint_pos'][:, 0:3]
    current_initial_pos = current_root_pos[0]
    print(f"  当前初始位置: {current_initial_pos}")
    
    # 计算偏移（将当前位置平移到demo的初始位置）
    # 但为了更合理，我们使用当前第一帧的位置作为参考，只平移XY，保持Z
    # 或者直接平移到原点附近
    offset = current_initial_pos.copy()
    # 将初始位置平移到demo的初始位置附近（但保持Z高度）
    target_initial_pos = np.array([demo_initial_pos[0], demo_initial_pos[1], current_initial_pos[2]])
    offset_to_apply = current_initial_pos - target_initial_pos
    
    print(f"\n[步骤3] 计算位置偏移")
    print(f"  目标初始位置: {target_initial_pos}")
    print(f"  需要应用的偏移: {offset_to_apply}")
    
    # 应用偏移到joint_pos的root位置
    print(f"\n[步骤4] 应用位置偏移")
    fixed_data = {}
    for key in data.keys():
        arr = data[key]
        if key == 'joint_pos':
            # 修复root位置（前3个元素是xyz）
            fixed_joint_pos = arr.copy()
            fixed_joint_pos[:, 0:3] = arr[:, 0:3] - offset_to_apply
            fixed_data[key] = fixed_joint_pos
            print(f"  修复 {key}: root位置已平移")
        elif key == 'body_pos_w':
            # 修复所有body位置
            fixed_body_pos = arr.copy()
            fixed_body_pos[:, :, :] = arr[:, :, :] - offset_to_apply
            fixed_data[key] = fixed_body_pos
            print(f"  修复 {key}: 所有body位置已平移")
        else:
            # 其他数据保持不变
            fixed_data[key] = arr
    
    # 验证修复后的位置
    print(f"\n[步骤5] 验证修复后的位置")
    fixed_root_pos = fixed_data['joint_pos'][:, 0:3]
    print(f"  修复后初始位置: {fixed_root_pos[0]}")
    print(f"  修复后位置范围: X=[{fixed_root_pos[:, 0].min():.3f}, {fixed_root_pos[:, 0].max():.3f}], "
          f"Y=[{fixed_root_pos[:, 1].min():.3f}, {fixed_root_pos[:, 1].max():.3f}], "
          f"Z=[{fixed_root_pos[:, 2].min():.3f}, {fixed_root_pos[:, 2].max():.3f}]")
    
    # 对比demo参数
    print(f"\n[步骤6] 参数对比")
    print(f"\n  【位置对比】")
    print(f"    Demo Root位置范围: X=[{demo_root_pos[:, 0].min():.3f}, {demo_root_pos[:, 0].max():.3f}], "
          f"Y=[{demo_root_pos[:, 1].min():.3f}, {demo_root_pos[:, 1].max():.3f}], "
          f"Z=[{demo_root_pos[:, 2].min():.3f}, {demo_root_pos[:, 2].max():.3f}]")
    print(f"    当前 Root位置范围: X=[{fixed_root_pos[:, 0].min():.3f}, {fixed_root_pos[:, 0].max():.3f}], "
          f"Y=[{fixed_root_pos[:, 1].min():.3f}, {fixed_root_pos[:, 1].max():.3f}], "
          f"Z=[{fixed_root_pos[:, 2].min():.3f}, {fixed_root_pos[:, 2].max():.3f}]")
    
    # 速度对比
    demo_joint_vel = demo_data['joint_vel']
    demo_root_lin_vel = demo_joint_vel[:, 0:3]
    demo_dof_vel = demo_joint_vel[:, 6:]
    demo_body_lin_vel = demo_data['body_lin_vel_w']
    
    current_joint_vel = fixed_data['joint_vel']
    current_root_lin_vel = current_joint_vel[:, 0:3]
    current_dof_vel = current_joint_vel[:, 6:]
    current_body_lin_vel = fixed_data['body_lin_vel_w']
    
    print(f"\n  【速度对比】")
    print(f"    Demo Root线性速度: 最大={np.max(np.linalg.norm(demo_root_lin_vel, axis=-1)):.4f} m/s")
    print(f"    当前 Root线性速度: 最大={np.max(np.linalg.norm(current_root_lin_vel, axis=-1)):.4f} m/s")
    print(f"    Demo DOF速度: 最大={np.max(np.abs(demo_dof_vel)):.4f} rad/s")
    print(f"    当前 DOF速度: 最大={np.max(np.abs(current_dof_vel)):.4f} rad/s")
    print(f"    Demo Body线性速度: 最大={np.max(np.linalg.norm(demo_body_lin_vel, axis=-1)):.4f} m/s")
    print(f"    当前 Body线性速度: 最大={np.max(np.linalg.norm(current_body_lin_vel, axis=-1)):.4f} m/s")
    
    # 检查是否有不合理的地方
    print(f"\n[步骤7] 参数合理性检查")
    warnings = []
    
    # 检查DOF位置范围
    demo_dof_pos = demo_data['joint_pos'][:, 7:]
    current_dof_pos = fixed_data['joint_pos'][:, 7:]
    if current_dof_pos.max() > 3.0 or current_dof_pos.min() < -3.0:
        warnings.append(f"⚠️  DOF位置范围较大: [{current_dof_pos.min():.3f}, {current_dof_pos.max():.3f}] rad")
    
    # 检查速度是否太小
    if np.max(np.linalg.norm(current_root_lin_vel, axis=-1)) < 0.5:
        warnings.append(f"⚠️  Root线性速度较小: {np.max(np.linalg.norm(current_root_lin_vel, axis=-1)):.4f} m/s (Demo: {np.max(np.linalg.norm(demo_root_lin_vel, axis=-1)):.4f} m/s)")
    
    if np.max(np.abs(current_dof_vel)) < 5.0:
        warnings.append(f"⚠️  DOF速度较小: {np.max(np.abs(current_dof_vel)):.4f} rad/s (Demo: {np.max(np.abs(demo_dof_vel)):.4f} rad/s)")
    
    if warnings:
        print("  发现以下问题:")
        for w in warnings:
            print(f"    {w}")
    else:
        print("  ✅ 所有参数在合理范围内")
    
    # 保存修复后的文件
    print(f"\n[步骤8] 保存修复后的文件: {output_file}")
    save_data = {}
    for k, v in fixed_data.items():
        if hasattr(v, 'dtype'):
            if v.dtype != object and 'float' in str(v.dtype):
                save_data[k] = v.astype(np.float32)
            else:
                save_data[k] = v
        else:
            save_data[k] = v
    np.savez(output_file, **save_data)
    
    print(f"\n✅ 修复完成！")
    print(f"文件位置: {output_file}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("用法: python fix_motion_to_demo_standard.py <输入文件> <输出文件> [参考demo文件]")
        print("示例: python fix_motion_to_demo_standard.py input.npz output.npz")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    reference_demo_file = sys.argv[3] if len(sys.argv) > 3 else None
    
    fix_motion_to_demo_standard(input_file, output_file, reference_demo_file)
