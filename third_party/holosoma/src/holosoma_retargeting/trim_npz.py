#!/usr/bin/env python3
"""
剪切 npz 动作文件的工具
"""
import numpy as np
import sys
from pathlib import Path

def trim_npz(input_file, output_file, start_frame=0, end_frame=None):
    """
    剪切 npz 文件
    
    Args:
        input_file: 输入的 npz 文件路径
        output_file: 输出的 npz 文件路径
        start_frame: 起始帧（包含），从 0 开始
        end_frame: 结束帧（不包含），None 表示到文件末尾
    """
    print(f"加载文件: {input_file}")
    data = np.load(input_file, allow_pickle=True)
    
    # 检查文件格式
    if 'joint_pos' in data:
        total_frames = data['joint_pos'].shape[0]
        print(f"总帧数: {total_frames}")
    else:
        print("警告: 未找到 joint_pos，尝试其他格式...")
        total_frames = None
    
    if end_frame is None:
        end_frame = total_frames if total_frames else len(list(data.values())[0])
    
    print(f"剪切范围: 第 {start_frame} 帧 到 第 {end_frame} 帧 (共 {end_frame - start_frame} 帧)")
    
    # 验证范围
    if start_frame < 0:
        raise ValueError(f"起始帧不能小于 0: {start_frame}")
    if end_frame > total_frames:
        raise ValueError(f"结束帧 {end_frame} 超过总帧数 {total_frames}")
    if start_frame >= end_frame:
        raise ValueError(f"起始帧 {start_frame} 必须小于结束帧 {end_frame}")
    
    # 需要保持不变的非时间序列数据键
    non_temporal_keys = {'joint_names', 'body_names', 'fps'}
    
    # 获取所有需要剪切的数组
    trimmed_data = {}
    
    for key in data.keys():
        arr = data[key]
        
        # 如果是元数据（非时间序列），保持不变
        if key in non_temporal_keys:
            trimmed_data[key] = arr
            print(f"  {key}: 保持不变 {arr.shape if hasattr(arr, 'shape') else type(arr)}")
        # 如果是时间序列数据（第一维等于总帧数），进行剪切
        elif isinstance(arr, np.ndarray) and len(arr.shape) > 0:
            if arr.shape[0] == total_frames:  # 第一维是时间维度
                trimmed_data[key] = arr[start_frame:end_frame]
                print(f"  {key}: {arr.shape} -> {trimmed_data[key].shape}")
            else:
                # 第一维不是时间维度，保持不变
                trimmed_data[key] = arr
                print(f"  {key}: 保持不变 {arr.shape}")
        else:
            # 标量或其他类型，保持不变
            trimmed_data[key] = arr
            print(f"  {key}: 保持不变 (标量)")
    
    # 保存剪切后的文件
    print(f"\n保存到: {output_file}")
    np.savez(output_file, **trimmed_data)
    
    if 'joint_pos' in trimmed_data:
        print(f"✓ 剪切完成！原始 {total_frames} 帧 -> 剪切后 {trimmed_data['joint_pos'].shape[0]} 帧")

if __name__ == "__main__":
    # 直接使用用户指定的参数
    input_file = "converted_res/robot_only/fight1_subject2_mj_fps50.npz"
    output_file = "converted_res/robot_only/fight1_subject2_trimmed_10700_11700.npz"
    start_frame = 10700
    end_frame = 11700
    
    trim_npz(input_file, output_file, start_frame, end_frame)
