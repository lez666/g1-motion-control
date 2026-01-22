#!/usr/bin/env python3
"""
移除开头的插值帧，并将XY坐标平移到原点（用于可视化）
"""
import numpy as np
import argparse
from pathlib import Path

def process_for_visualization(input_file, output_file, start_interp_frames=14):
    """
    移除开头插值帧，并将XY坐标平移到原点
    
    Args:
        input_file: 输入文件（包含开头插值的文件）
        output_file: 输出文件（处理后的文件）
        start_interp_frames: 要移除的开头插值帧数
    """
    print(f"加载文件: {input_file}")
    data = np.load(input_file, allow_pickle=True)

    print(f"原始文件总帧数: {data['joint_pos'].shape[0]}")

    # 1. 移除开头的插值帧
    trimmed_data = {}
    for key in data.keys():
        if key in ['joint_names', 'body_names', 'fps']:
            trimmed_data[key] = data[key]
        elif isinstance(data[key], np.ndarray) and len(data[key].shape) > 0:
            if data[key].shape[0] == data['joint_pos'].shape[0]:
                trimmed_data[key] = data[key][start_interp_frames:]
            else:
                trimmed_data[key] = data[key]
        else:
            trimmed_data[key] = data[key]
    
    print(f"移除开头插值后总帧数: {trimmed_data['joint_pos'].shape[0]}")

    # 2. 只平移x,y到原点，z保持原值
    first_root_pos = trimmed_data['joint_pos'][0, :3].copy()
    print(f"\n第一帧root位置（原始）: {first_root_pos}")

    trimmed_data['joint_pos'][:, 0:2] -= first_root_pos[:2]  # 只平移x,y
    trimmed_data['body_pos_w'][:, :, 0:2] -= first_root_pos[:2]  # body的x,y也平移
    
    print(f"平移x,y后第一帧root位置: {trimmed_data['joint_pos'][0, :3]}")

    # 3. 调整z使双脚踩地
    body_names = trimmed_data['body_names'].tolist()
    foot_body_names = [name for name in body_names if 'ankle' in name.lower() or 'foot' in name.lower()]
    print(f"\n所有body名称（前10个）: {body_names[:10]}")
    print(f"找到的脚部body: {foot_body_names}")

    foot_body_indices = [body_names.index(name) for name in foot_body_names if name in body_names]
    if foot_body_indices:
        foot_z_values = trimmed_data['body_pos_w'][:, foot_body_indices, 2]  # (T, n_foot_bodies)
        min_foot_z_per_frame = np.min(foot_z_values, axis=1)  # (T,) 每帧的最低脚部z值
        global_min_foot_z = np.min(min_foot_z_per_frame)  # 所有帧中的最低脚部z值
        
        print(f"最低脚部z值（所有帧）: {global_min_foot_z}")
        
        # 调整所有帧的root z位置，使得最低脚部z值在地面上（z=0）
        z_offset = -global_min_foot_z
        trimmed_data['joint_pos'][:, 2] += z_offset
        trimmed_data['body_pos_w'][:, :, 2] += z_offset
        
        print(f"调整z偏移: {z_offset}")
        print(f"调整后最低脚部z值: {np.min(trimmed_data['body_pos_w'][:, foot_body_indices, 2])}")
    else:
        print("警告: 未找到脚部body，使用root位置调整")
        min_root_z = np.min(trimmed_data['joint_pos'][:, 2])
        z_offset = -min_root_z
        trimmed_data['joint_pos'][:, 2] += z_offset
        trimmed_data['body_pos_w'][:, :, 2] += z_offset

    # 保存可视化文件
    np.savez(output_file, **trimmed_data)
    print(f"\n✓ 保存可视化文件: {output_file}")
    print(f"  总帧数: {trimmed_data['joint_pos'].shape[0]}")
    print(f"  第一帧root位置: {trimmed_data['joint_pos'][0, :3]}")
    print(f"  最后一帧root位置: {trimmed_data['joint_pos'][-1, :3]}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="处理NPZ用于可视化: 移除开头插值，平移XY到原点，调整Z使脚踩地。")
    parser.add_argument("--input_file", type=str, required=True, help="输入的NPZ文件路径（包含开头插值）。")
    parser.add_argument("--output_file", type=str, required=True, help="保存处理后的NPZ文件路径（用于可视化）。")
    parser.add_argument("--start_interp_frames", type=int, default=None, help="要移除的开头插值帧数（如果不指定，会根据FPS自动计算0.25s）。")
    args = parser.parse_args()

    # 如果没有指定帧数，自动计算0.25s对应的帧数
    if args.start_interp_frames is None:
        data = np.load(args.input_file, allow_pickle=True)
        fps = float(np.array(data["fps"]).reshape(-1)[0])
        args.start_interp_frames = int(fps * 0.25)
        print(f"自动计算: FPS={fps}, 0.25s = {args.start_interp_frames} 帧")

    process_for_visualization(args.input_file, args.output_file, args.start_interp_frames)
