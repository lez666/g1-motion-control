# G1机器人站立走路控制器训练配置生成任务 (Holosoma 原生模式)

## 项目背景
这是一个基于 **Holosoma** 框架的 G1 人形机器人强化学习训练项目。Holosoma 是一个高度模块化的框架，它将训练配置解耦为 `Experiment`, `Robot`, `Simulator`, `Command`, `Reward`, `Observation`, `Randomization`, `Termination`, `Curriculum` 等多个子模块。

你需要按照 Holosoma 的原生架构，生成完整的 G1 机器人站立走路训练配置文件。

## 技术架构要求

### 1. 目录结构与文件组织
Holosoma 的配置采用层级化的 Python 类系统。对于 G1 机器人的 locomotion 任务，代码应组织在以下路径：
`src/holosoma/holosoma/config_values/loco/g1/`
- `experiment.py`: 核心入口，定义 `ExperimentConfig` 并绑定所有子配置。
- `reward.py`: 定义 `RewardManagerCfg` 和奖励项 `RewardTermCfg`。
- `observation.py`: 定义 `ObservationManagerCfg`。
- `command.py`: 定义 `CommandManagerCfg`。
- `randomization.py`: 定义 `RandomizationManagerCfg` (域随机化)。
- `termination.py`: 定义 `TerminationManagerCfg`。
- `curriculum.py`: 定义 `CurriculumManagerCfg`。
- `action.py`: 定义 `ActionManagerCfg`。

### 2. 核心类定义参考
所有配置必须继承自 `holosoma.config_types` 中的对应类。

#### ExperimentConfig (experiment.py) 示例：
```python
from holosoma.config_types.experiment import ExperimentConfig, TrainingConfig
from holosoma.config_values import algo, robot, simulator, terrain
from . import action, command, curriculum, observation, randomization, reward, termination

g1_29dof_loco_custom = ExperimentConfig(
    env_class="holosoma.envs.locomotion.locomotion_manager.LeggedRobotLocomotionManager",
    training=TrainingConfig(project="g1-locomotion", name="g1_custom_run"),
    algo=algo.ppo, # 或 algo.fast_sac
    simulator=simulator.isaacsim,
    robot=robot.g1_29dof,
    terrain=terrain.terrain_locomotion_plane,
    observation=observation.g1_29dof_loco_custom,
    action=action.g1_29dof_joint_pos,
    termination=termination.g1_29dof_loco_custom,
    randomization=randomization.g1_29dof_loco_custom,
    command=command.g1_29dof_loco_custom,
    curriculum=curriculum.g1_29dof_loco_custom,
    reward=reward.g1_29dof_loco_custom,
)
```

## 具体任务需求

### 1. 机器人基础配置 (由 `robot.g1_29dof` 提供)
- **初始姿态**: 
  - 位置: (0.0, 0.0, 0.74)
  - 关键关节角度: `.*_hip_pitch_joint`: -0.20, `.*_knee_joint`: 0.42, `.*_ankle_pitch_joint`: -0.23, `.*_elbow_joint`: 0.87.

### 2. 奖励函数配置 (`reward.py`)
需要定义 `g1_29dof_loco_custom = RewardManagerCfg(...)`。包含以下项：
- **Velocity Tracking**: `tracking_lin_vel` (weight: 2.0), `tracking_ang_vel` (weight: 2.0)。
- **Gait Quality**: `feet_air_time` (weight: 0.1), `both_feet_air` (weight: -0.3), `feet_slide` (weight: -0.1)。
- **Posture**: `flat_orientation_l2` (weight: -1.0), `joint_deviation_l1` (weight: -0.1)。
- **Smoothness**: `penalty_action_rate` (weight: -0.01), `joint_torques_l2` (weight: -1e-4)。
- **Safety**: `joint_pos_limits` (weight: -1.0)。
- **Survival**: `termination` penalty (weight: -200.0)。

### 3. 观察空间配置 (`observation.py`)
定义 `ObservationManagerCfg`，包含 `actor_obs` 组：
- `projected_gravity`, `velocity_commands`, `joint_pos_rel`, `joint_vel_rel`, `actions`。
- 注意：必须指定 `history_length`（通常为 1-3）。

### 4. 域随机化配置 (`randomization.py`)
- `physics_material_randomization`: 摩擦力 (0.4, 1.0)。
- `randomize_body_mass`: 缩放系数 (0.9, 1.1)。
- `add_torso_mass`: 对 `torso_link` 添加 (-1.0, 1.0) kg 负载。
- `push_robot`: 间隔随机推力。

### 5. 命令配置 (`command.py`)
- 使用 `UniformVelocityCommandCfg`。
- 线速度 X: (-1.0, 1.0), Y: (-0.5, 0.5)。
- 角速度 Yaw: (-1.0, 1.0)。

## 输出要求
请生成符合 Holosoma 规范的 Python 模块化代码。
1. **代码注释**: 使用中文说明关键超参数的物理意义。
2. **导入语句**: 必须包含完整的 `holosoma.config_types` 和 `holosoma.config_values` 导入。
3. **Preset 命名**: 将你的实验命名为 `g1_29dof_loco_refined` 并确保其在 `experiment.py` 的 `__all__` 中。
4. **训练执行**: 提醒用户使用 `python src/holosoma/holosoma/train_agent.py exp:g1-29dof-loco-refined` 启动训练。

---
**注意**: 这是 Holosoma 原生 Demo 模式，不要混合使用普通的 Isaac Lab 脚本。所有逻辑应通过 `ExperimentConfig` 的子模块实例化完成。
