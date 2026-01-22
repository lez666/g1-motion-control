#!/usr/bin/env python3
"""
清理 .npz motion 文件中的 NaN 和 Inf 值
使用前向填充（forward fill）或线性插值替换
"""
import numpy as np
import argparse
from pathlib import Path

def clean_nan_inf(npz_path, output_path, method='forward_fill'):
    """
    清理 NaN 和 Inf
    
    Args:
        npz_path: 输入文件
        output_path: 输出文件
        method: 清理方法
            - 'forward_fill': 用前一个有效值填充
            - 'linear': 线性插值
            - 'zero': 用0填充
    """
    print(f"加载文件: {npz_path}")
    data = np.load(npz_path, allow_pickle=True)
    
    cleaned_data = {}
    total_fixed = 0
    
    for key in data.keys():
        arr = data[key]
        
        # 非数组字段直接复制
        if not isinstance(arr, np.ndarray) or key in ['joint_names', 'body_names', 'fps']:
            cleaned_data[key] = arr
            continue
        
        arr_cleaned = arr.copy()
        
        # 检查NaN和Inf
        nan_mask = np.isnan(arr_cleaned)
        inf_mask = np.isinf(arr_cleaned)
        invalid_mask = nan_mask | inf_mask
        invalid_count = invalid_mask.sum()
        
        if invalid_count > 0:
            print(f"清理 {key}: 发现 {invalid_count} 个无效值")
            total_fixed += invalid_count
            
            if method == 'forward_fill':
                # 前向填充：用前一个有效值填充（按时间维度）
                if arr_cleaned.ndim >= 1:
                    # 对每一帧，如果无效，用前一个有效帧的值填充
                    for t in range(arr_cleaned.shape[0]):
                        frame_invalid = invalid_mask[t, ...] if arr_cleaned.ndim > 1 else invalid_mask[t]
                        if frame_invalid.any() if arr_cleaned.ndim > 1 else frame_invalid:
                            # 找到前一个有效帧
                            for prev_t in range(t-1, -1, -1):
                                prev_valid = ~invalid_mask[prev_t, ...] if arr_cleaned.ndim > 1 else ~invalid_mask[prev_t]
                                if prev_valid.all() if arr_cleaned.ndim > 1 else prev_valid:
                                    if arr_cleaned.ndim > 1:
                                        arr_cleaned[t, ...] = np.where(
                                            invalid_mask[t, ...],
                                            arr_cleaned[prev_t, ...],
                                            arr_cleaned[t, ...]
                                        )
                                    else:
                                        arr_cleaned[t] = arr_cleaned[prev_t]
                                    break
                            else:
                                # 如果前面没有有效值，用0填充
                                if arr_cleaned.ndim > 1:
                                    arr_cleaned[t, ...] = np.where(invalid_mask[t, ...], 0.0, arr_cleaned[t, ...])
                                else:
                                    arr_cleaned[t] = 0.0
                else:
                    arr_cleaned[invalid_mask] = 0.0
                    
            elif method == 'linear':
                # 线性插值（只对时间维度）
                if arr_cleaned.ndim >= 1:
                    for t in range(arr_cleaned.shape[0]):
                        frame_invalid = invalid_mask[t, ...] if arr_cleaned.ndim > 1 else invalid_mask[t]
                        if frame_invalid.any() if arr_cleaned.ndim > 1 else frame_invalid:
                            # 找到前后有效值进行插值
                            prev_t = None
                            next_t = None
                            for i in range(t-1, -1, -1):
                                if not invalid_mask[i, ...].any() if arr_cleaned.ndim > 1 else not invalid_mask[i]:
                                    prev_t = i
                                    break
                            for i in range(t+1, arr_cleaned.shape[0]):
                                if not invalid_mask[i, ...].any() if arr_cleaned.ndim > 1 else not invalid_mask[i]:
                                    next_t = i
                                    break
                            
                            if prev_t is not None and next_t is not None:
                                # 线性插值
                                alpha = (t - prev_t) / (next_t - prev_t)
                                if arr_cleaned.ndim > 1:
                                    arr_cleaned[t, ...] = (1 - alpha) * arr_cleaned[prev_t, ...] + alpha * arr_cleaned[next_t, ...]
                                else:
                                    arr_cleaned[t] = (1 - alpha) * arr_cleaned[prev_t] + alpha * arr_cleaned[next_t]
                            elif prev_t is not None:
                                # 只有前一个有效值，复制它
                                if arr_cleaned.ndim > 1:
                                    arr_cleaned[t, ...] = arr_cleaned[prev_t, ...]
                                else:
                                    arr_cleaned[t] = arr_cleaned[prev_t]
                            elif next_t is not None:
                                # 只有后一个有效值，复制它
                                if arr_cleaned.ndim > 1:
                                    arr_cleaned[t, ...] = arr_cleaned[next_t, ...]
                                else:
                                    arr_cleaned[t] = arr_cleaned[next_t]
                            else:
                                # 没有有效值，用0填充
                                if arr_cleaned.ndim > 1:
                                    arr_cleaned[t, ...] = 0.0
                                else:
                                    arr_cleaned[t] = 0.0
                else:
                    arr_cleaned[invalid_mask] = 0.0
                    
            elif method == 'zero':
                # 用0填充
                arr_cleaned[invalid_mask] = 0.0
            
            # 验证清理结果
            remaining_nan = np.isnan(arr_cleaned).sum()
            remaining_inf = np.isinf(arr_cleaned).sum()
            if remaining_nan > 0 or remaining_inf > 0:
                print(f"  警告: 仍有 {remaining_nan} 个NaN和 {remaining_inf} 个Inf，用0填充")
                arr_cleaned = np.nan_to_num(arr_cleaned, nan=0.0, posinf=0.0, neginf=0.0)
        else:
            print(f"✓ {key}: 无需清理")
        
        cleaned_data[key] = arr_cleaned
    
    # 保存清理后的文件
    print(f"\n保存清理后的文件: {output_path}")
    np.savez(output_path, **cleaned_data)
    
    print(f"✓ 完成！共修复 {total_fixed} 个无效值")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="清理 .npz 文件中的 NaN 和 Inf")
    parser.add_argument("--input", type=str, required=True, help="输入的 .npz 文件路径")
    parser.add_argument("--output", type=str, required=True, help="输出的 .npz 文件路径")
    parser.add_argument("--method", type=str, default="forward_fill", 
                       choices=['forward_fill', 'linear', 'zero'],
                       help="清理方法: forward_fill (前向填充), linear (线性插值), zero (用0填充)")
    args = parser.parse_args()
    
    clean_nan_inf(args.input, args.output, args.method)
