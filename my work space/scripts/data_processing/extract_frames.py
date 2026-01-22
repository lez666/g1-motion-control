#!/usr/bin/env python3
"""
从npz文件中截取指定帧范围并保存为新文件
"""
import numpy as np
import sys
from pathlib import Path

def extract_frames(input_file, output_file, start_frame, end_frame):
    """
    从npz文件中截取指定帧范围
    
    Args:
        input_file: 输入的npz文件路径
        output_file: 输出的npz文件路径
        start_frame: 起始帧（包含）
        end_frame: 结束帧（包含）
    """
    print(f"加载文件: {input_file}")
    data = np.load(input_file, allow_pickle=True)
    
    # 获取总帧数（从第一个数组字段推断）
    total_frames = None
    for key in data.keys():
        arr = data[key]
        if hasattr(arr, 'shape') and len(arr.shape) > 0:
            total_frames = arr.shape[0]
            break
    
    if total_frames is None:
        raise ValueError("无法确定总帧数")
    
    print(f"总帧数: {total_frames}")
    print(f"截取范围: [{start_frame}, {end_frame}] (共 {end_frame - start_frame + 1} 帧)")
    
    # 验证范围
    if start_frame < 0:
        start_frame = 0
        print(f"警告: 起始帧调整为 0")
    if end_frame >= total_frames:
        end_frame = total_frames - 1
        print(f"警告: 结束帧调整为 {end_frame}")
    
    if start_frame > end_frame:
        raise ValueError(f"起始帧 ({start_frame}) 不能大于结束帧 ({end_frame})")
    
    # 提取数据
    extracted_data = {}
    for key in data.keys():
        arr = data[key]
        
        if hasattr(arr, 'shape'):
            # 如果是数组，检查第一个维度是否是时间维度
            if len(arr.shape) > 0 and arr.shape[0] == total_frames:
                # 第一个维度是时间维度，进行切片
                if len(arr.shape) == 1:
                    extracted_data[key] = arr[start_frame:end_frame+1]
                elif len(arr.shape) == 2:
                    extracted_data[key] = arr[start_frame:end_frame+1, :]
                elif len(arr.shape) == 3:
                    extracted_data[key] = arr[start_frame:end_frame+1, :, :]
                else:
                    # 对于更高维度，只切片第一个维度
                    slices = [slice(start_frame, end_frame+1)] + [slice(None)] * (len(arr.shape) - 1)
                    extracted_data[key] = arr[tuple(slices)]
                print(f"  截取 {key}: {arr.shape} -> {extracted_data[key].shape}")
            else:
                # 不是时间维度，直接复制
                extracted_data[key] = arr
                print(f"  保留 {key}: {arr.shape if hasattr(arr, 'shape') else type(arr)}")
        else:
            # 非数组数据，直接复制
            extracted_data[key] = arr
            print(f"  保留 {key}: {type(arr)}")
    
    # 保存新文件
    print(f"\n保存到: {output_file}")
    np.savez(output_file, **extracted_data)
    
    # 验证新文件
    print("\n验证新文件...")
    verify_data = np.load(output_file, allow_pickle=True)
    verify_frames = None
    for key in verify_data.keys():
        arr = verify_data[key]
        if hasattr(arr, 'shape') and len(arr.shape) > 0:
            verify_frames = arr.shape[0]
            break
    
    print(f"✅ 提取完成！")
    print(f"  新文件帧数: {verify_frames}")
    print(f"  文件大小: {Path(output_file).stat().st_size / (1024*1024):.2f} MB")

if __name__ == "__main__":
    if len(sys.argv) < 5:
        print("用法: python extract_frames.py <输入文件> <输出文件> <起始帧> <结束帧>")
        print("示例: python extract_frames.py input.npz output.npz 2000 4300")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    start_frame = int(sys.argv[3])
    end_frame = int(sys.argv[4])
    
    extract_frames(input_file, output_file, start_frame, end_frame)
