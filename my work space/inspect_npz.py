#!/usr/bin/env python3
import numpy as np

data = np.load('fight1_subject3_robot_motion.npz', allow_pickle=True)

print("=== 文件内容检查 ===")
for k in data.keys():
    val = data[k]
    if hasattr(val, 'shape'):
        print(f"{k}: shape={val.shape}, dtype={val.dtype}")
        if val.dtype == object:
            print(f"  -> 是 object 类型，内容: {val.item() if val.size == 1 else 'array'}")
            if val.size == 1:
                item = val.item()
                if isinstance(item, (list, np.ndarray)):
                    print(f"  -> 实际内容类型: {type(item)}, 长度: {len(item) if hasattr(item, '__len__') else 'N/A'}")
                    if isinstance(item, list) and len(item) > 0:
                        print(f"  -> 第一个元素: {item[0]}")
                    elif isinstance(item, np.ndarray):
                        print(f"  -> 数组形状: {item.shape}")
    else:
        print(f"{k}: {type(val)} = {val}")

print("\n=== 详细检查 object 类型字段 ===")
if 'link_body_list' in data:
    link_list = data['link_body_list']
    if link_list.dtype == object and link_list.size == 1:
        link_list_val = link_list.item()
        print(f"link_body_list: {link_list_val}")
        if isinstance(link_list_val, (list, np.ndarray)):
            print(f"  长度: {len(link_list_val)}")
            print(f"  前5个: {link_list_val[:5] if len(link_list_val) >= 5 else link_list_val}")

if 'local_body_pos' in data:
    body_pos = data['local_body_pos']
    if body_pos.dtype == object and body_pos.size == 1:
        body_pos_val = body_pos.item()
        print(f"local_body_pos 类型: {type(body_pos_val)}")
        if isinstance(body_pos_val, np.ndarray):
            print(f"  shape: {body_pos_val.shape}")
            print(f"  dtype: {body_pos_val.dtype}")
        elif isinstance(body_pos_val, list):
            print(f"  长度: {len(body_pos_val)}")
            if len(body_pos_val) > 0:
                print(f"  第一个元素类型: {type(body_pos_val[0])}")
                if isinstance(body_pos_val[0], np.ndarray):
                    print(f"  第一个元素 shape: {body_pos_val[0].shape}")
