# G1 Motion Control 🤖

Humanoid motion control and reinforcement learning for Unitree G1.

## ⚠️ Prerequisites

This project requires a high-performance Ubuntu workstation. **You MUST ensure the base `holosoma` framework is fully configured before proceeding.**

- **NVIDIA GPU** (RTX 3090/4090 recommended)
- **NVIDIA Drivers** & **CUDA Toolkit** (12.x recommended)
- **Python 3.10+** (Conda environment highly recommended)
- **Holosoma Environment**: Verify that you can run basic Holosoma examples first.

---

## 🚀 Quick Start

### 1. Clone and Basic Dependencies
```bash
git clone --recursive <repo-url>
cd g1-motion-control
./scripts/bootstrap.sh  # Sync submodules and install control dependencies
```

### 2. Full Holosoma Setup
Navigate to the submodule directory and complete the environment initialization according to the official workflow:
```bash
cd third_party/holosoma/scripts

# Option A: Full IsaacSim Installation (Required for training)
./setup_isaacsim.sh

# Option B: Full MuJoCo Installation (For fast simulation inference)
./setup_mujoco.sh

# Option C: Inference Environment (For run_multi_policy_sim2sim.py)
./setup_inference.sh
```
*Note: If you encounter permission or path issues, please refer to `third_party/holosoma/README.md`.*

### 3. Training
```bash
# Activate IsaacSim environment and start training
cd third_party/holosoma
source scripts/source_isaacsim_setup.sh
python src/holosoma/holosoma/train_agent.py \
    exp:g1-29dof-robust \
    reward:g1-29dof-loco-robust-refined \
    --training.num-envs 8192
```

---

## 🎮 Simulation & Real-time Control (MuJoCo)

### Step A: Start Simulator
```bash
cd third_party/holosoma && source scripts/source_mujoco_setup.sh
python src/holosoma/holosoma/run_sim.py robot:g1-29dof terrain:terrain_locomotion_plane
```

### Step B: Run Controller (Keyboard Support)
```bash
cd third_party/holosoma && source scripts/source_inference_setup.sh
python3 "../my work space/run_multi_policy_sim2sim.py" <path_to_onnx>
```

---

## ⌨️ Keyboard Controls

1. **MuJoCo Window**: Press `8` to lower gantry, `9` to remove gantry.
2. **Controller Terminal**: Press `]` to activate policy.
3. **Mode Switch**: Number keys `1` (Stand), `2` (Walk).
4. **Real-time Movement**:
   - `↑ ↓ ← →`: Move (Forward/Back/Left/Right)
   - `Q / E`: Rotate (Left/Right)
   - `Z`: Zero velocity

---

## 📦 Pre-trained Models
- **`model_22200.onnx`**: Latest refined locomotion (Stable gait & Upright posture).
- **`model_39999.onnx`**: WBT policy for crawling and motion tracking.

## 📁 Structure
- `configs/`: G1 configurations
- `my work space/`: Inference scripts & training logs
- `scripts/`: Utility scripts
- `third_party/holosoma/`: Core framework
