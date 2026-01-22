#!/usr/bin/env python3
"""
修复motion文件中的四元数归一化问题
"""
import numpy as np
import argparse

def fix_quaternion_normalization(input_file, output_file):
    """
    修复四元数归一化问题
    
    Args:
        input_file: 输入文件
        output_file: 输出文件
    """
    print(f"加载文件: {input_file}")
    data = np.load(input_file, allow_pickle=True)

    # 检查并修复四元数
    body_quat_w = data['body_quat_w'].copy()
    print(f"\n修复前四元数模长: min={np.linalg.norm(body_quat_w, axis=-1).min():.6f}, max={np.linalg.norm(body_quat_w, axis=-1).max():.6f}")

    # 计算每个四元数的模长
    norms = np.linalg.norm(body_quat_w, axis=-1)  # (T, B)
    
    # 对于模长为0的四元数，替换为单位四元数 [1, 0, 0, 0] (wxyz格式)
    zero_mask = norms < 1e-6  # (T, B)
    zero_count = zero_mask.sum()
    
    if zero_count > 0:
        print(f"发现 {zero_count} 个零四元数，将替换为单位四元数")
        # 对于零四元数，设置为单位四元数 [1, 0, 0, 0] (wxyz格式)
        body_quat_w[zero_mask] = np.array([1.0, 0.0, 0.0, 0.0])
        norms[zero_mask] = 1.0

    # 归一化所有四元数
    body_quat_w = body_quat_w / norms[..., np.newaxis]  # 广播归一化

    print(f"修复后四元数模长: min={np.linalg.norm(body_quat_w, axis=-1).min():.6f}, max={np.linalg.norm(body_quat_w, axis=-1).max():.6f}, mean={np.linalg.norm(body_quat_w, axis=-1).mean():.6f}")

    # 保存修复后的文件
    output_data = {}
    for key in data.keys():
        if key == 'body_quat_w':
            output_data[key] = body_quat_w
        else:
            output_data[key] = data[key]

    print(f"\n保存修复后的文件: {output_file}")
    np.savez(output_file, **output_data)

    print("✓ 完成！")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="修复motion文件中的四元数归一化问题")
    parser.add_argument("--input", type=str, required=True, help="输入的NPZ文件路径")
    parser.add_argument("--output", type=str, required=True, help="输出的NPZ文件路径")
    args = parser.parse_args()
    
    fix_quaternion_normalization(args.input, args.output)
