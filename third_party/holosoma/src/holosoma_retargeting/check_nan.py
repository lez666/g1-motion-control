#!/usr/bin/env python3
"""
检查 .npz motion 文件中是否有 NaN 或 Inf 值
"""
import numpy as np
import argparse
from pathlib import Path

def check_nan_inf(npz_path):
    """检查 .npz 文件中的 NaN 和 Inf"""
    print(f"检查文件: {npz_path}")
    data = np.load(npz_path, allow_pickle=True)
    
    has_issues = False
    total_nan = 0
    total_inf = 0
    
    for key in data.keys():
        arr = data[key]
        if isinstance(arr, np.ndarray):
            # 跳过字符串数组
            if arr.dtype.kind in ['U', 'S', 'O']:  # Unicode, String, Object
                print(f"  {key}: {type(arr)} (字符串数组，跳过检查)")
                continue
            nan_count = np.isnan(arr).sum()
            inf_count = np.isinf(arr).sum()
            
            if nan_count > 0 or inf_count > 0:
                has_issues = True
                total_nan += nan_count
                total_inf += inf_count
                print(f"⚠️  {key}:")
                print(f"   形状: {arr.shape}")
                print(f"   NaN数量: {nan_count}")
                print(f"   Inf数量: {inf_count}")
                
                # 显示NaN/Inf的位置
                if nan_count > 0:
                    nan_indices = np.where(np.isnan(arr))
                    if len(nan_indices[0]) > 0:
                        print(f"   NaN位置（前5个）: {list(zip(*[idx[:5] for idx in nan_indices]))}")
                if inf_count > 0:
                    inf_indices = np.where(np.isinf(arr))
                    if len(inf_indices[0]) > 0:
                        print(f"   Inf位置（前5个）: {list(zip(*[idx[:5] for idx in inf_indices]))}")
            else:
                print(f"✓ {key}: 正常 (形状: {arr.shape})")
        else:
            print(f"  {key}: {type(arr)} (非数组)")
    
    print("\n" + "="*50)
    if has_issues:
        print(f"❌ 发现问题！")
        print(f"   总NaN数量: {total_nan}")
        print(f"   总Inf数量: {total_inf}")
        return False
    else:
        print(f"✓ 文件正常，没有NaN或Inf")
        return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="检查 .npz 文件中的 NaN 和 Inf")
    parser.add_argument("--npz_path", type=str, required=True, help="要检查的 .npz 文件路径")
    args = parser.parse_args()
    
    check_nan_inf(args.npz_path)
