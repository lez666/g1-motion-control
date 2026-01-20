# G1机器人站立走路控制器训练配置生成任务

## 项目背景
这是一个基于Isaac Lab和RSL-RL的G1人形机器人强化学习训练项目。需要生成完整的站立走路控制器训练配置文件，用于在holosoma环境下进行训练。

## 技术要求

### 1. 机器人模型配置
- **模型格式**: USD文件（不是URDF）
- **模型路径**: "urdf_file="g1/g1_29dof.urdf",  # 使用URDF  你可以按照holosoma项目里转成usd
- **站立初始状态**:
  - 位置: (0.0, 0.0, 0.74) 米
  - 关节初始角度:
    - `.*_hip_pitch_joint`: -0.20 rad
    - `.*_knee_joint`: 0.42 rad
    - `.*_ankle_pitch_joint`: -0.23 rad
    - `.*_elbow_joint`: 0.87 rad
    - `left_shoulder_roll_joint`: 0.16 rad
    - `left_shoulder_pitch_joint`: 0.35 rad
    - `right_shoulder_roll_joint`: -0.16 rad
    - `right_shoulder_pitch_joint`: 0.35 rad

### 2. 执行器配置
需要配置4组执行器：
- **legs**: hip_yaw, hip_roll, hip_pitch, knee (使用7520系列电机参数)
- **feet**: ankle_pitch, ankle_roll (使用5020系列电机参数，stiffness和damping为2倍)
- **waist_yaw**: waist_yaw_joint (使用7520_14参数)
- **arms**: 所有shoulder、elbow、wrist关节 (使用5020系列电机参数)

执行器参数计算：
- NATURAL_FREQ = 10 * 2.0 * π ≈ 62.83 rad/s
- DAMPING_RATIO = 2.0
- ARMATURE_5020 = 0.003609725
- ARMATURE_7520_14 = 0.010177520
- ARMATURE_7520_22 = 0.025101925
- STIFFNESS = ARMATURE * NATURAL_FREQ²
- DAMPING = 2.0 * DAMPING_RATIO * ARMATURE * NATURAL_FREQ

**具体执行器配置**：

**legs执行器组**:
- 关节: `.*_hip_yaw_joint`, `.*_hip_roll_joint`, `.*_hip_pitch_joint`, `.*_knee_joint`
- 力矩限制:
  - `.*_hip_yaw_joint`: 88.0 N·m
  - `.*_hip_roll_joint`: 139.0 N·m
  - `.*_hip_pitch_joint`: 88.0 N·m
  - `.*_knee_joint`: 139.0 N·m
- 速度限制:
  - `.*_hip_yaw_joint`: 32.0 rad/s
  - `.*_hip_roll_joint`: 20.0 rad/s
  - `.*_hip_pitch_joint`: 32.0 rad/s
  - `.*_knee_joint`: 20.0 rad/s
- 刚度:
  - `.*_hip_pitch_joint`: STIFFNESS_7520_14
  - `.*_hip_roll_joint`: STIFFNESS_7520_22
  - `.*_hip_yaw_joint`: STIFFNESS_7520_14
  - `.*_knee_joint`: STIFFNESS_7520_22
- 阻尼:
  - `.*_hip_pitch_joint`: DAMPING_7520_14
  - `.*_hip_roll_joint`: DAMPING_7520_22
  - `.*_hip_yaw_joint`: DAMPING_7520_14
  - `.*_knee_joint`: DAMPING_7520_22
- 惯量:
  - `.*_hip_pitch_joint`: ARMATURE_7520_14
  - `.*_hip_roll_joint`: ARMATURE_7520_22
  - `.*_hip_yaw_joint`: ARMATURE_7520_14
  - `.*_knee_joint`: ARMATURE_7520_22

**feet执行器组**:
- 关节: `.*_ankle_pitch_joint`, `.*_ankle_roll_joint`
- 力矩限制: 50.0 N·m
- 速度限制: 37.0 rad/s
- 刚度: 2.0 * STIFFNESS_5020
- 阻尼: 2.0 * DAMPING_5020
- 惯量: 2.0 * ARMATURE_5020

**waist_yaw执行器组**:
- 关节: `waist_yaw_joint`
- 力矩限制: 88.0 N·m
- 速度限制: 32.0 rad/s
- 刚度: STIFFNESS_7520_14
- 阻尼: DAMPING_7520_14
- 惯量: ARMATURE_7520_14

**arms执行器组**:
- 关节: `.*_shoulder_pitch_joint`, `.*_shoulder_roll_joint`, `.*_shoulder_yaw_joint`, `.*_elbow_joint`, `.*_wrist_roll_joint`
- 力矩限制: 25.0 N·m (所有关节)
- 速度限制: 37.0 rad/s (所有关节)
- 刚度: STIFFNESS_5020 (所有关节)
- 阻尼: DAMPING_5020 (所有关节)
- 惯量: ARMATURE_5020 (所有关节)

### 3. 环境配置
- **环境数量**: 8192
- **环境间距**: 4.0
- **仿真时间步**: 0.005秒
- **降采样**: 4 (控制频率 = 50Hz)
- **回合长度**: 20秒
- **地形**: 平面地形（terrain_type = "plane"）
- **物理材质**: 
  - 静态摩擦: 1.0
  - 动态摩擦: 1.0
  - 恢复系数: 0.0

### 4. 观察空间配置
**Policy观察组**:
- `projected_gravity`: 投影重力向量（带噪声 ±0.05）
- `velocity_commands`: 速度命令（x, y, yaw）
- `joint_pos_rel`: 相对关节位置（带噪声 ±0.01）
- `joint_vel_rel`: 相对关节速度（带噪声 ±1.5）
- `actions`: 上一步动作

**Critic观察组**: 与Policy相同，但不加噪声

### 5. 动作空间配置
- **类型**: JointPositionAction
- **缩放**: 0.5
- **使用默认偏移**: True
- **关节名称**: `.*` (所有关节)

### 6. 命令配置
**速度命令** (`base_velocity`):
- **重采样时间范围**: (3.0, 8.0) 秒
- **速度范围**:
  - `lin_vel_x`: (-1.0, 1.0) m/s
  - `lin_vel_y`: (-0.5, 0.5) m/s
  - `ang_vel_z`: (-1.0, 1.0) rad/s
  - `heading`: (-π, π) rad
- **相对站立环境**: 0.1
- **相对航向环境**: 1.0
- **航向控制**: False
- **航向控制刚度**: 0.5
- **调试可视化**: True

### 7. 奖励函数配置（必须完全按照以下配置）

```python
RewardsCfg:
  # 速度跟踪奖励
  track_lin_vel_xy_exp:
    func: track_lin_vel_xy_yaw_frame_exp
    weight: 2.0
    params:
      command_name: "base_velocity"
      std: 0.5
  
  track_ang_vel_z_exp:
    func: track_ang_vel_z_world_exp
    weight: 2.0
    params:
      command_name: "base_velocity"
      std: 0.5
  
  # 姿态奖励
  flat_orientation_l2:
    func: flat_orientation_l2
    weight: -1.0
  
  # 步态质量奖励
  feet_air_time:
    func: feet_air_time_positive_biped
    weight: 0.1
    params:
      command_name: "base_velocity"
      sensor_cfg: SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link")
      threshold: 0.2
  
  both_feet_air:
    func: both_feet_air
    weight: -0.3
    params:
      sensor_cfg: SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link")
  
  feet_slide:
    func: feet_slide
    weight: -0.1
    params:
      sensor_cfg: SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link")
      asset_cfg: SceneEntityCfg("robot", body_names=".*_ankle_roll_link")
  
  # 关节限制惩罚
  dof_torques_l2:
    func: joint_torques_l2
    weight: -1.0e-4
    params:
      asset_cfg: SceneEntityCfg("robot", joint_names=[".*_hip_.*", ".*_knee_joint", ".*_ankle_pitch_joint", ".*_ankle_roll_joint"])
  
  dof_acc_l2:
    func: joint_acc_l2
    weight: -2.5e-7
    params:
      asset_cfg: SceneEntityCfg("robot", joint_names=[".*_hip_.*", ".*_knee_joint", ".*_ankle_pitch_joint", ".*_ankle_roll_joint"])
  
  action_rate_l2:
    func: action_rate_l2
    weight: -0.01
  
  dof_pos_limits:
    func: joint_pos_limits
    weight: -1.0
    params:
      asset_cfg: SceneEntityCfg("robot", joint_names=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"])
  
  joint_deviation_all:
    func: joint_deviation_l1
    weight: -0.1
    params:
      asset_cfg: SceneEntityCfg("robot")
  
  # 终止惩罚
  termination_penalty:
    func: is_terminated
    weight: -200.0
```

### 8. 终止条件配置
- `time_out`: 超时终止（time_out=True）
- `base_contact`: 非法接触终止
  - 传感器: `contact_forces`
  - 身体名称: `^(?!.*ankle_roll_link).*` (除了脚踝滚转连杆外的所有身体)
  - 阈值: 1.0 N

### 9. 事件配置（域随机化）
**启动时事件**:
- `physics_material`: 随机化刚体材料
  - 静态摩擦范围: (0.4, 1.0)
  - 动态摩擦范围: (0.4, 1.0)
  - 恢复系数范围: (0.0, 0.0)
  - 桶数: 64
  
- `randomize_body_mass`: 随机化身体质量
  - 操作: "scale"
  - 质量分布参数: (0.9, 1.1)
  - 应用到所有身体: `.*`
  
- `add_torso_mass`: 添加躯干质量
  - 操作: "add"
  - 质量分布参数: (-1.0, 1.0) kg
  - 仅应用到: `torso_link`
  
- `randomize_joint_friction`: 随机化关节摩擦
  - 操作: "scale"
  - 摩擦分布参数: (0.5, 2.0)
  - 应用到所有关节: `.*`
  
- `randomize_joint_armature`: 随机化关节惯量
  - 操作: "scale"
  - 惯量分布参数: (1.0, 1.05)
  - 应用到所有关节: `.*`

**重置时事件**:
- `reset_base`: 随机化根状态
  - 位置范围: x(-0.5, 0.5), y(-0.5, 0.5), yaw(-π, π)
  - 速度范围: 
    - 线性: x(-0.5, 0.5), y(-0.5, 0.5), z(-0.5, 0.5) m/s
    - 角速度: roll(-0.5, 0.5), pitch(-0.5, 0.5), yaw(-0.5, 0.5) rad/s
  
- `reset_robot_joints`: 随机化关节位置
  - 位置范围: (0.5, 1.5) 倍默认位置
  - 速度范围: (0.0, 0.0) rad/s

**间隔事件**:
- `push_robot`: 随机推动机器人
  - 模式: "interval"
  - 间隔范围: (2.0, 4.0) 秒
  - 速度范围: x(-0.5, 0.5), y(-0.5, 0.5) m/s

### 10. PPO算法配置
```python
PPORunnerCfg:
  num_steps_per_env: 24
  max_iterations: 1000
  save_interval: 250
  experiment_name: "g1_locomotion"
  empirical_normalization: False
  clip_actions: 5.0
  
  policy:
    init_noise_std: 1.0
    actor_hidden_dims: [256, 128, 128]
    critic_hidden_dims: [256, 128, 128]
    activation: "elu"
  
  algorithm:
    value_loss_coef: 1.0
    use_clipped_value_loss: True
    clip_param: 0.05
    entropy_coef: 0.008
    num_learning_epochs: 5
    num_mini_batches: 4
    learning_rate: 1.0e-3
    schedule: "adaptive"
    gamma: 0.99
    lam: 0.95
    desired_kl: 0.01
    max_grad_norm: 1.0
    symmetry_cfg:
      use_data_augmentation: True
      use_mirror_loss: True
      mirror_loss_coeff: 0.5
      # 需要实现G1机器人的对称性数据增强函数
      # 左右对称：交换左右腿和左右臂的关节
```

### 11. 传感器配置
- `contact_forces`: 接触力传感器
  - 路径: `{ENV_REGEX_NS}/Robot/.*`
  - 历史长度: 3
  - 跟踪空中时间: True
  - 更新周期: 0.005秒（与仿真时间步一致）

### 12. 场景配置
- **地面**: 使用TerrainImporter配置平面地形
- **光照**: 使用DomeLight，强度750.0
- **机器人**: 使用G1_STAND_CFG配置，路径为`{ENV_REGEX_NS}/Robot`
- **物理设置**:
  - GPU最大刚体补丁数: 10 * 2^15
  - 位置迭代次数: 8
  - 速度迭代次数: 4
  - 禁用自碰撞: True

### 13. 文件结构要求
需要生成以下文件：
1. **环境配置文件**: `g1_locomotion_env_cfg.py`
   - 包含场景配置、观察配置、动作配置、命令配置、奖励配置、终止配置、事件配置
   - 使用`@configclass`装饰器
   - 继承自`ManagerBasedRLEnvCfg`

2. **PPO配置文件**: `rsl_rl_ppo_cfg.py`
   - 包含PPO算法超参数配置
   - 继承自`RslRlOnPolicyRunnerCfg`
   - 包含对称性配置

3. **机器人模型配置**: `g1.py`
   - 包含G1_STAND_CFG定义
   - 包含执行器参数计算
   - 包含初始状态配置

4. **对称性函数**: `symmetry_func.py` (如果需要)
   - 实现G1机器人的左右对称数据增强函数
   - 交换左右腿和左右臂的关节顺序

### 14. 代码风格要求
- 使用Isaac Lab的配置类系统（`@configclass`装饰器）
- 使用`RewTerm`, `ObsTerm`, `DoneTerm`, `EventTerm`等配置类
- 使用`SceneEntityCfg`指定场景实体
- 遵循Isaac Lab的命名约定
- 所有配置必须类型安全
- 添加必要的导入语句
- 添加代码注释说明关键参数

### 15. 关键注意事项
1. **不要使用URDF**，使用USD格式模型文件
2. **奖励权重必须精确匹配**，这些权重经过调优
3. **执行器参数必须正确计算**，使用给定的ARMATURE值和公式
4. **观察空间必须包含所有列出的项**，顺序很重要
5. **域随机化配置必须完整**，这对sim-to-real迁移至关重要
6. **对称性增强必须启用**，这对双足机器人很重要
7. **关节名称使用正则表达式匹配**，如`.*_hip_pitch_joint`匹配所有髋关节
8. **初始状态配置必须准确**，影响训练起点
9. **传感器配置必须正确**，接触力传感器对步态学习至关重要
10. **命令配置的频率和范围要合理**，影响训练难度

### 16. 训练命令示例
训练时使用以下命令格式：
```bash
python scripts/rsl_rl/train.py \
  --task g1-locomotion \
  --agent rsl_rl_ppo_cfg \
  --num_envs 4096 \
  --max_iterations 1000
```

### 17. 输出要求
请生成完整的、可直接运行的训练配置文件，包括：
1. 完整的环境配置类（继承自ManagerBasedRLEnvCfg）
2. 完整的PPO算法配置类（继承自RslRlOnPolicyRunnerCfg）
3. 机器人模型配置类（ArticulationCfg）
4. 必要的导入语句
5. 代码注释说明关键参数
6. 对称性数据增强函数（如果使用）

所有代码必须遵循Isaac Lab的最佳实践，并且可以直接用于在holosoma环境下进行训练。

### 18. 验证检查清单
生成代码后，请确保：
- [ ] 所有执行器参数计算正确
- [ ] 奖励函数权重与配置一致
- [ ] 观察空间包含所有必需项
- [ ] 域随机化配置完整
- [ ] 终止条件配置正确
- [ ] 命令配置合理
- [ ] 传感器配置正确
- [ ] 代码可以正常导入
- [ ] 配置类继承关系正确
- [ ] 所有必要的导入语句已包含
