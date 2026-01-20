# G1 Motion Control 🤖

Humanoid motion control and reinforcement learning for Unitree G1.

## 🚀 快速开始 (Quick Start)

### 1. 环境配置 (Setup)
```bash
git clone --recursive <repo-url>
cd g1-motion-control
./scripts/bootstrap.sh
```

### 2. 训练命令 (Training - IsaacSim)
```bash
cd third_party/holosoma
source scripts/source_isaacsim_setup.sh

# 推荐：使用 8192 环境进行训练 (Recommended: 8192 envs)
python src/holosoma/holosoma/train_agent.py \
    exp:g1-29dof-robust \
    reward:g1-29dof-loco-robust-refined \
    --training.num-envs 8192

# 从检查点继续训练 (Resume training)
python src/holosoma/holosoma/train_agent.py \
    exp:g1-29dof-robust \
    --training.checkpoint <path_to_model_xxxx.pt>
```

---

## 🎮 仿真备忘录 (Simulation Cheat Sheet - MuJoCo)

### 终端 A：启动仿真环境 (Start Simulator)
**默认平地 (Default Plane):**
```bash
cd third_party/holosoma && source scripts/source_mujoco_setup.sh
python src/holosoma/holosoma/run_sim.py robot:g1-29dof terrain:terrain_locomotion_plane
```

**加载斜坡 (Load Slope):**
```bash
cd third_party/holosoma && source scripts/source_mujoco_setup.sh
python src/holosoma/holosoma/run_sim.py robot:g1-29dof terrain:terrain_load_obj \
    --terrain.terrain-term.obj-file-path="src/holosoma/holosoma/data/motions/g1_29dof/whole_body_tracking/terrain_slope.obj"
```

### 终端 B：运行策略控制 (Run Policy)
**1. Locomotion 策略 (支持方向键实时控制):**
```bash
cd third_party/holosoma && source scripts/source_inference_setup.sh
# 运行您训练的模型 (Run your trained model)
python3 "../my work space/run_multi_policy_sim2sim.py" <path_to_onnx>
```

**2. WBT 策略 (跳舞/爬行):**
```bash
cd third_party/holosoma && source scripts/source_inference_setup.sh
python3 src/holosoma_inference/holosoma_inference/run_policy.py inference:g1-29dof-wbt \
    --task.model-path <WBT_ONNX_PATH> \
    --task.no-use-joystick \
    --task.interface lo
```

---

## ⌨️ 操作要点 (Operations)

1. **MuJoCo 窗口**: 按 `8` 降低吊架，按 `9` 移除吊架。
2. **控制终端**: 按 `]` 激活策略 (Activate Policy)。
3. **模式切换**: 数字键 `1` (站立模式 Stand), `2` (走路模式 Walk)。
4. **实时运动控制** (仅限走路模式):
   - `↑ ↓ ← →`: 前进、后退、左平移、右平移
   - `Q / E`: 左转、右转
   - `Z`: 速度清零 (Zero velocity)

---

## 📦 Pre-trained Models
- **`model_22200.onnx`**: Latest refined locomotion (Stable gait & Upright posture).
- **`model_39999.onnx`**: WBT policy for crawling and motion tracking.
- **Legacy**: `model_04600.onnx` and `model_03300.onnx` are kept for reference.

## 📁 结构 (Structure)
- `configs/`: G1 configurations
- `my work space/`: Inference scripts & training logs
- `scripts/`: Utility scripts
- `third_party/holosoma/`: Core framework
