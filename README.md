# G1 Motion Control 🤖

Humanoid motion control and reinforcement learning for Unitree G1.

## ⚠️ Prerequisites (重要前置要求)

This project requires a high-performance Ubuntu workstation. **You MUST ensure the base `holosoma` framework is fully configured before proceeding.**

- **NVIDIA GPU** (RTX 3090/4090 recommended)
- **NVIDIA Drivers** & **CUDA Toolkit** (12.x recommended)
- **Python 3.10+** (Conda environment highly recommended)
- **Holosoma Environment**: Verify that you can run basic Holosoma examples first.

---

## 🚀 快速开始 (Quick Start)

### 1. 代码获取与基本依赖
```bash
git clone --recursive <repo-url>
cd g1-motion-control
./scripts/bootstrap.sh  # 同步子模块并安装本项目特定的控制依赖
```

### 2. 仿真框架完全配置 (Full Holosoma Setup)
进入子模块目录，按照官方流程完成完整的环境初始化：
```bash
cd third_party/holosoma/scripts

# 选项 A: 完整安装 IsaacSim (必须完成，用于训练)
./setup_isaacsim.sh

# 选项 B: 完整安装 MuJoCo (用于快速仿真推理)
./setup_mujoco.sh

# 选项 C: 安装推理环境 (用于 run_multi_policy_sim2sim.py)
./setup_inference.sh
```
*提示：如果遇到权限或路径问题，请参考 `third_party/holosoma/README.md`。*

### 3. 训练命令 (Training)
```bash
# 激活 IsaacSim 环境并开始训练
cd third_party/holosoma
source scripts/source_isaacsim_setup.sh
python src/holosoma/holosoma/train_agent.py \
    exp:g1-29dof-robust \
    reward:g1-29dof-loco-robust-refined \
    --training.num-envs 8192
```

---

## 🎮 仿真与实时控制 (Simulation - MuJoCo)

### 步骤 A：启动仿真器
```bash
cd third_party/holosoma && source scripts/source_mujoco_setup.sh
python src/holosoma/holosoma/run_sim.py robot:g1-29dof terrain:terrain_locomotion_plane
```

### 步骤 B：运行控制脚本 (支持键盘方向键)
```bash
cd third_party/holosoma && source scripts/source_inference_setup.sh
python3 "../my work space/run_multi_policy_sim2sim.py" <path_to_onnx>
```

---

## ⌨️ 键盘控制指南 (Keyboard Controls)

1. **MuJoCo 窗口**: 按 `8` 降吊架, 按 `9` 卸吊架 (Gantry control)。
2. **控制终端**: 按 `]` 激活策略 (Activate Policy)。
3. **模式切换**: 数字键 `1` (站立 Stand), `2` (走路 Walk)。
4. **实时运动**: `↑ ↓ ← →` (移动), `Q / E` (旋转), `Z` (归零)。

---

## 📦 Pre-trained Models
- **`model_22200.onnx`**: Latest refined locomotion (Stable gait & Upright posture).
- **`model_39999.onnx`**: WBT policy for crawling and motion tracking.

## 📁 结构 (Structure)
- `configs/`: G1 configurations
- `my work space/`: Inference scripts & training logs
- `scripts/`: Utility scripts
- `third_party/holosoma/`: Core framework
