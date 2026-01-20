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
git clone <repo-url>
cd g1-motion-control
./scripts/bootstrap.sh  # Install control dependencies
```

### 2. Full Holosoma Setup
Holosoma is now integrated into this repository as a local library. Navigate to the directory and complete the environment initialization:
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
- `my work space/`: Inference scripts & training logs (Contains selected ONNX models)
- `scripts/`: Utility scripts
- `third_party/`:
  - `holosoma/`: Core framework (Locally customized, not synced to upstream)
  - `beyond_mimic/`: [Future] Motion imitation framework
  - `gr00t/`: [Future] NVIDIA GR00T foundation model integration
  - `twist2/`: [Future] Locomotion control
  - `gmr/`: [Future] Gaussian Mixture Regression/Robotics
  - `common_assets/`: Shared robot URDFs and meshes

---

## 🛠️ Third-Party Integration Plan (Roadmap)

To maintain a clean and manageable workspace while integrating multiple complex repositories, we follow these principles:

### 1. Vendor Strategy (No Upstream Sync)
All `third_party` projects are "vendored" into this repository. This means:
- We do **not** use Git Submodules for active development.
- Local modifications are encouraged and committed directly to the main repository.
- Version history of the original repo is removed to keep the workspace lightweight.

### 2. Environment Isolation
Each major framework (`gr00t`, `holosoma`, `beyond_mimic`) may require conflicting dependency versions (Isaac Sim 2023 vs 4.0, different PyTorch versions, etc.).
- **Rule**: Use separate Conda environments or Docker containers for each major integration.
- Environment-specific setup scripts should be placed in `scripts/envs/`.

### 3. Shared Robot Assets
Avoid duplicating large STL/Meshes across different `third_party` folders.
- Store the "Source of Truth" for G1 URDF/Meshes in `third_party/common_assets/`.
- Use soft links (`ln -s`) or config path overrides to point external frameworks to these shared assets.

### 4. Weights & Model Management
- **Git LFS**: Large `.pt` and `.onnx` files should be tracked via Git LFS or kept outside the main git history if they are intermediate training results.
- **Redundancy Cleanup**: Intermediate models are periodically pruned (keeping only start and final milestones) to save space.
