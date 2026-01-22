# 数据处理脚本

本目录包含用于处理motion数据的主要脚本。

## 主要脚本

### 1. convert_pkl_to_wbt.py
将.pkl文件转换为WBT训练格式的.npz文件，并自动应用速度缩放以匹配demo标准。

**用法:**
```bash
python convert_pkl_to_wbt.py \
    --pkl_path /path/to/input.pkl \
    --output_path /path/to/output.npz \
    --fps 50 \
    --reference_demo /path/to/demo.npz
```

### 2. extract_frames.py
从npz文件中提取指定帧范围。

**用法:**
```bash
python extract_frames.py input.npz output.npz 2000 4300
```

### 3. fix_motion_position.py
修复motion数据的位置偏移，使其符合demo标准（平移到原点附近）。

**用法:**
```bash
python fix_motion_position.py input.npz output.npz
```

## 依赖

这些脚本依赖于holosoma_retargeting模块，需要先激活环境：
```bash
cd /path/to/holosoma
source scripts/source_retargeting_setup.sh
```

## 注意事项

- 所有脚本都会自动匹配demo的速度和位置标准
- 输出文件使用50 FPS（WBT训练标准）
- 速度会自动缩放以匹配demo标准
