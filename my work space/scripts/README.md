# 脚本目录

本目录包含项目使用的各种脚本工具。

## 目录结构

- `data_processing/` - 数据处理脚本（转换、提取、修复motion数据）
- `utils/` - 工具脚本（分析、验证等）

## 快速开始

### 数据处理

```bash
# 转换pkl到WBT格式
cd scripts/data_processing
python convert_pkl_to_wbt.py --pkl_path input.pkl --output_path output.npz

# 提取帧范围
python extract_frames.py input.npz output.npz 2000 4300

# 修复位置偏移
python fix_motion_position.py input.npz output.npz
```

详细说明请参考各子目录的README。
