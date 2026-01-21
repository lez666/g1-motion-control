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

## 💡 DIY Training & Customization

Since `holosoma` is now a local library, you should place your custom logic within its package structure to ensure the dynamic loading system works correctly.

### 1. Where to put your code?

| Component | Target Path |
| :--- | :--- |
| **Reward Functions (Logic)** | `third_party/holosoma/src/holosoma/holosoma/managers/reward/terms/` |
| **Reward Configs (Weights)** | `third_party/holosoma/src/holosoma/holosoma/config_values/loco/g1/reward.py` |
| **Experiment Presets** | `third_party/holosoma/src/holosoma/holosoma/config_values/loco/g1/experiment.py` |
| **Environment/Obs/Action DIY** | `third_party/holosoma/src/holosoma/holosoma/config_values/loco/g1/` |

### 2. How to run your DIY experiment?

1. Define your reward functions in `managers/reward/terms/`.
2. Reference them in `config_values/loco/g1/reward.py` by creating a new `RewardManagerCfg`.
3. Create a new `ExperimentConfig` in `experiment.py` (e.g., `g1_29dof_diy`).
4. Start training:
   ```bash
   python src/holosoma/holosoma/train_agent.py exp:g1-29dof-diy
   ```

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

---

## 🎬 Motion Data Processing for Whole Body Tracking

### 1. Convert BVH to LAFAN Format

Convert BVH files to `.npy` format for retargeting:

```bash
cd third_party/holosoma/src/holosoma_retargeting
source ../../scripts/source_retargeting_setup.sh
cd data_utils
python extract_global_positions.py \
    --input_dir <bvh_directory> \
    --output_dir <output_directory>
```

### 2. Run Retargeting

Convert human motion to robot motion:

```bash
cd third_party/holosoma/src/holosoma_retargeting
source ../../scripts/source_retargeting_setup.sh
python examples/robot_retarget.py \
    --data_path <npy_directory> \
    --task-type robot_only \
    --task-name <sequence_name> \
    --data_format lafan \
    --task-config.ground-range -10 10 \
    --save_dir demo_results/g1/robot_only/<output_dir> \
    --retargeter.foot-sticking-tolerance 0.02
```

### 3. Convert to Training Format

Convert retargeted `.npz` to training format:

```bash
cd third_party/holosoma/src/holosoma_retargeting
source ../../scripts/source_retargeting_setup.sh
python data_conversion/convert_data_format_mj.py \
    --input_file <retargeted_file>.npz \
    --output_fps 50 \
    --output_name <output_file>.npz \
    --data_format lafan \
    --object_name "ground" \
    --once
```

### 4. Trim Motion File

Extract specific frame range from motion file. **Note**: Edit the script to set input/output paths and frame range:

```bash
cd third_party/holosoma/src/holosoma_retargeting
source ../../scripts/source_retargeting_setup.sh
# Edit trim_npz.py to set input_file, output_file, start_frame, end_frame
python trim_npz.py
```

Example script configuration:
```python
input_file = "converted_res/robot_only/motion.npz"
output_file = "converted_res/robot_only/motion_trimmed.npz"
start_frame = 10700
end_frame = 11700
```

### 5. Add Initial Pose Interpolation

Prepend interpolated frames from reference initial pose to motion start. **Note**: Edit the script to set file paths and interpolation duration:

```bash
python prepend_interpolation.py
```

Script configuration:
```python
reference_file = "converted_res/robot_only/original_motion.npz"  # Extract frame 0 as reference
input_file = "converted_res/robot_only/motion_trimmed.npz"  # Motion to prepend
output_file = "converted_res/robot_only/motion_with_interp.npz"
num_interp_frames = 13  # 0.25s at 50fps
reference_frame_idx = 0  # Use frame 0 from reference file
```

This script:
- Extracts initial pose from reference motion file (frame 0)
- Interpolates from reference pose to motion start
- Preserves target frame x,y position to avoid horizontal drift
- Only interpolates z position, orientation, and joint angles

### 6. Visualize Motion File

View motion sequence in browser:

```bash
cd third_party/holosoma/src/holosoma_retargeting
source ../../scripts/source_retargeting_setup.sh
python data_conversion/viser_body_vel_player.py \
    --npz_path <motion_file>.npz \
    --robot_urdf models/g1/g1_29dof.urdf
```

Open the displayed URL (usually `http://localhost:8080`) to:
- Playback motion sequence
- Control playback with slider
- View body positions and velocity vectors
- Toggle mesh and velocity arrow display

### 7. Train Whole Body Tracking

```bash
cd third_party/holosoma
source scripts/source_isaacsim_setup.sh
python src/holosoma/holosoma/train_agent.py \
    exp:g1-29dof-wbt \
    logger:wandb \
    --command.setup_terms.motion_command.params.motion_config.motion_file=<absolute_or_relative_path_to_motion_file>.npz
```

**Note**: 
- Use absolute path or path relative to project root for `motion_file`
- Training uses `enable_default_pose_prepend=True` by default, which adds 2s interpolation from config default pose to motion start
- If motion file already has initial pose interpolation, this provides smooth double transition
