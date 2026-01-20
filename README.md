# G1 Motion Control 🤖

Humanoid motion control and reinforcement learning for Unitree G1.

## 🚀 快速开始

### 1. 环境配置
```bash
git clone --recursive <repo-url>
cd g1-motion-control
./scripts/bootstrap.sh
```

### 2. 训练命令 (IsaacSim)
```bash
cd third_party/holosoma
source scripts/source_isaacsim_setup.sh

# 推荐：使用 8192 环境进行训练
python src/holosoma/holosoma/train_agent.py \
    exp:g1-29dof-robust \
    reward:g1-29dof-loco-robust-refined \
    --training.num-envs 8192
```

---

## 🎮 仿真与部署 (MuJoCo)

### 步骤 A：启动仿真环境
**平地地形 (默认):**
```bash
cd third_party/holosoma
source scripts/source_mujoco_setup.sh
python src/holosoma/holosoma/run_sim.py robot:g1-29dof terrain:terrain_locomotion_plane
```

**斜坡地形:**
```bash
cd third_party/holosoma
source scripts/source_mujoco_setup.sh
python src/holosoma/holosoma/run_sim.py robot:g1-29dof terrain:terrain_load_obj \
    --terrain.terrain-term.obj-file-path="src/holosoma/holosoma/data/motions/g1_29dof/whole_body_tracking/terrain_slope.obj"
```

### 步骤 B：运行策略控制
**Locomotion 策略 (支持方向键控制):**
```bash
cd third_party/holosoma
source scripts/source_inference_setup.sh
# 使用最新训练的 ONNX 模型
python3 "../my work space/run_multi_policy_sim2sim.py" <path_to_latest_onnx>
```

**WBT 策略 (特殊动作):**
- **跳舞:** 使用 `fastsac_g1_29dof_dancing.onnx`
- **爬行:** 使用 WBT 实验目录下的 `model_39999.onnx`

---

## ⌨️ 键盘控制指南

1. **初始化**: 在 MuJoCo 窗口按 `8` 降低吊架，按 `9` 移除吊架。
2. **启动**: 在控制终端按 `]` 启动策略。
3. **模式切换**: 
   - 按 `1`: 站立模式 (Stand)
   - 按 `2`: 走路模式 (Walk)
4. **运动控制** (仅限走路模式):
   - `↑ ↓ ← →`: 前进、后退、左移、右移
   - `Q / E`: 左转、右转
   - `Z`: 速度清零

## 📁 项目结构
- `configs/`: G1 机器人及奖励函数配置
- `my work space/`: 推理脚本、分析工具及训练日志
- `scripts/`: 项目引导与工具脚本
- `third_party/holosoma/`: 核心仿真与训练框架 (Submodule)
