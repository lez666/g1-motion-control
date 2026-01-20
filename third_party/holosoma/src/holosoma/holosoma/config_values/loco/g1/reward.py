"""Locomotion reward presets for the G1 robot."""

from holosoma.config_types.reward import RewardManagerCfg, RewardTermCfg

g1_29dof_loco = RewardManagerCfg(
    only_positive_rewards=False,
    terms={
        "tracking_lin_vel": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:tracking_lin_vel",
            weight=2.0,
            params={"tracking_sigma": 0.25},
        ),
        "tracking_ang_vel": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:tracking_ang_vel",
            weight=1.5,
            params={"tracking_sigma": 0.25},
        ),
        "penalty_ang_vel_xy": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:penalty_ang_vel_xy",
            weight=-1.0,
            params={},
            tags=["penalty_curriculum"],
        ),
        "penalty_orientation": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:penalty_orientation",
            weight=-10.0,
            params={},
            tags=["penalty_curriculum"],
        ),
        "penalty_action_rate": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:penalty_action_rate",
            weight=-2.0,
            params={},
            tags=["penalty_curriculum"],
        ),
        "feet_phase": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:feet_phase",
            weight=5.0,
            params={"swing_height": 0.09, "tracking_sigma": 0.008},
        ),
        "pose": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:pose",
            weight=-0.5,
            params={
                "pose_weights": [
                    0.01,
                    1.0,
                    5.0,
                    0.01,
                    5.0,
                    5.0,
                    0.01,
                    1.0,
                    5.0,
                    0.01,
                    5.0,
                    5.0,
                    50.0,
                    50.0,
                    50.0,
                    50.0,
                    50.0,
                    50.0,
                    50.0,
                    50.0,
                    50.0,
                    50.0,
                    50.0,
                    50.0,
                    50.0,
                    50.0,
                    50.0,
                    50.0,
                    50.0,
                ],
            },
            tags=["penalty_curriculum"],
        ),
        "penalty_close_feet_xy": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:penalty_close_feet_xy",
            weight=-10.0,
            params={"close_feet_threshold": 0.15},
            tags=["penalty_curriculum"],
        ),
        "penalty_feet_ori": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:penalty_feet_ori",
            weight=-5.0,
            params={},
            tags=["penalty_curriculum"],
        ),
        "alive": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:alive",
            weight=1.0,
            params={},
        ),
    },
)

g1_29dof_loco_fast_sac = RewardManagerCfg(
    only_positive_rewards=False,
    terms={
        "tracking_lin_vel": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:tracking_lin_vel",
            weight=2.0,
            params={"tracking_sigma": 0.25},
        ),
        "tracking_ang_vel": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:tracking_ang_vel",
            weight=1.5,
            params={"tracking_sigma": 0.25},
        ),
        "penalty_ang_vel_xy": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:penalty_ang_vel_xy",
            weight=-1.0,
            params={},
            tags=["penalty_curriculum"],
        ),
        "penalty_orientation": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:penalty_orientation",
            weight=-10.0,
            params={},
            tags=["penalty_curriculum"],
        ),
        "penalty_action_rate": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:penalty_action_rate",
            weight=-2.0,
            params={},
            tags=["penalty_curriculum"],
        ),
        "feet_phase": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:feet_phase",
            weight=5.0,
            params={"swing_height": 0.09, "tracking_sigma": 0.008},
        ),
        "pose": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:pose",
            weight=-0.5,
            params={
                "pose_weights": [
                    0.01,
                    1.0,
                    5.0,
                    0.01,
                    5.0,
                    5.0,
                    0.01,
                    1.0,
                    5.0,
                    0.01,
                    5.0,
                    5.0,
                    50.0,
                    50.0,
                    50.0,
                    50.0,
                    50.0,
                    50.0,
                    50.0,
                    50.0,
                    50.0,
                    50.0,
                    50.0,
                    50.0,
                    50.0,
                    50.0,
                    50.0,
                    50.0,
                    50.0,
                ],
            },
            tags=["penalty_curriculum"],
        ),
        "penalty_close_feet_xy": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:penalty_close_feet_xy",
            weight=-10.0,
            params={"close_feet_threshold": 0.15},
            tags=["penalty_curriculum"],
        ),
        "penalty_feet_ori": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:penalty_feet_ori",
            weight=-5.0,
            params={},
            tags=["penalty_curriculum"],
        ),
        "alive": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:alive",
            weight=10.0,
            params={},
        ),
    },
)

g1_29dof_loco_robust = RewardManagerCfg(
    only_positive_rewards=False,
    terms={
        # Velocity tracking rewards (increased weights for better tracking)
        "tracking_lin_vel": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:tracking_lin_vel",
            weight=2.0,
            params={"tracking_sigma": 0.5},  # Increased from 0.25 to match prompt file
        ),
        "tracking_ang_vel": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:tracking_ang_vel",
            weight=2.0,  # Increased from 1.5 to match prompt file
            params={"tracking_sigma": 0.5},  # Increased from 0.25 to match prompt file
        ),
        # Orientation rewards
        "flat_orientation_l2": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:flat_orientation_l2",
            weight=-1.0,
            params={},
        ),
        "penalty_orientation": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:penalty_orientation",
            weight=-10.0,
            params={},
            tags=["penalty_curriculum"],
        ),
        # Gait quality rewards
        "feet_phase": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:feet_phase",
            weight=5.0,
            params={"swing_height": 0.09, "tracking_sigma": 0.008},
        ),
        "feet_air_time": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:feet_air_time_positive_biped",
            weight=0.1,
            params={"threshold": 0.2},
        ),
        "both_feet_air": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:both_feet_air",
            weight=-0.3,
            params={"threshold": 0.2},
        ),
        "feet_slide": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:feet_slide",
            weight=-0.1,
            params={"threshold": 0.2},
        ),
        # Joint limit penalties
        "dof_torques_l2": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:joint_torques_l2",
            weight=-1.0e-4,
            params={
                "joint_names": [".*_hip_.*", ".*_knee_joint", ".*_ankle_pitch_joint", ".*_ankle_roll_joint"]
            },
        ),
        "dof_acc_l2": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:joint_acc_l2",
            weight=-2.5e-7,
            params={
                "joint_names": [".*_hip_.*", ".*_knee_joint", ".*_ankle_pitch_joint", ".*_ankle_roll_joint"]
            },
        ),
        "dof_pos_limits": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:joint_pos_limits",
            weight=-1.0,
            params={"joint_names": [".*_ankle_pitch_joint", ".*_ankle_roll_joint"]},
        ),
        # Action smoothness
        "action_rate_l2": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:penalty_action_rate",
            weight=-0.01,  # Adjusted from -2.0 to match prompt file
            params={},
        ),
        # Joint deviation penalty
        "joint_deviation_all": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:joint_deviation_l1",
            weight=-0.1,
            params={},
        ),
        # Additional penalties
        "penalty_ang_vel_xy": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:penalty_ang_vel_xy",
            weight=-1.0,
            params={},
            tags=["penalty_curriculum"],
        ),
        "penalty_close_feet_xy": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:penalty_close_feet_xy",
            weight=-10.0,
            params={"close_feet_threshold": 0.15},
            tags=["penalty_curriculum"],
        ),
        "penalty_feet_ori": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:penalty_feet_ori",
            weight=-5.0,
            params={},
            tags=["penalty_curriculum"],
        ),
        # Survival reward (increased weight to encourage standing)
        "alive": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:alive",
            weight=2.0,  # Increased from 1.0 to encourage better stability
            params={},
        ),
        # Termination penalty (from Unitree RL Lab - critical for learning to stand)
        "termination_penalty": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:termination",
            weight=-200.0,  # Strong penalty for falling, encourages standing
            params={},
        ),
    },
)

g1_29dof_loco_robust_refined = RewardManagerCfg(
    only_positive_rewards=False,
    terms={
        # Velocity tracking rewards (increased weights for better tracking)
        "tracking_lin_vel": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:tracking_lin_vel",
            weight=2.0,
            params={"tracking_sigma": 0.5},
        ),
        "tracking_ang_vel": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:tracking_ang_vel",
            weight=2.0,
            params={"tracking_sigma": 0.5},
        ),
        # Orientation rewards
        "flat_orientation_l2": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:flat_orientation_l2",
            weight=-1.0,
            params={},
        ),
        "penalty_orientation": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:penalty_orientation",
            weight=-10.0,
            params={},
            tags=["penalty_curriculum"],
        ),
        # Gait quality rewards
        "feet_phase": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:feet_phase",
            weight=5.0,
            params={"swing_height": 0.09, "tracking_sigma": 0.008},
        ),
        "feet_air_time": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:feet_air_time_positive_biped",
            weight=0.1,
            params={"threshold": 0.2},
        ),
        "both_feet_air": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:both_feet_air",
            weight=-0.3,
            params={"threshold": 0.2},
        ),
        "feet_slide": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:feet_slide",
            weight=-0.1,
            params={"threshold": 0.2},
        ),
        # Joint limit penalties
        "dof_torques_l2": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:joint_torques_l2",
            weight=-1.0e-4,
            params={
                "joint_names": [".*_hip_.*", ".*_knee_joint", ".*_ankle_pitch_joint", ".*_ankle_roll_joint"]
            },
        ),
        "dof_acc_l2": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:joint_acc_l2",
            weight=-2.5e-7,
            params={
                "joint_names": [".*_hip_.*", ".*_knee_joint", ".*_ankle_pitch_joint", ".*_ankle_roll_joint"]
            },
        ),
        "dof_pos_limits": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:joint_pos_limits",
            weight=-1.0,
            params={"joint_names": [".*_ankle_pitch_joint", ".*_ankle_roll_joint"]},
        ),
        # Action smoothness
        "action_rate_l2": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:penalty_action_rate",
            weight=-0.01,
            params={},
        ),
        # Joint deviation penalty
        "joint_deviation_all": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:joint_deviation_l1",
            weight=-0.1,
            params={},
        ),
        # Additional penalties
        "penalty_ang_vel_xy": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:penalty_ang_vel_xy",
            weight=-1.0,
            params={},
            tags=["penalty_curriculum"],
        ),
        "penalty_close_feet_xy": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:penalty_close_feet_xy",
            weight=-10.0,
            params={"close_feet_threshold": 0.15},
            tags=["penalty_curriculum"],
        ),
        "penalty_feet_ori": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:penalty_feet_ori",
            weight=-5.0,
            params={},
            tags=["penalty_curriculum"],
        ),
        # Survival reward (increased weight to encourage standing)
        "alive": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:alive",
            weight=2.0,
            params={},
        ),
        # Termination penalty (reduced from -200.0 to -100.0 for less harsh penalty)
        "termination_penalty": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:termination",
            weight=-100.0,  # Reduced from -200.0 to allow more exploration
            params={},
        ),
        # Pose reward - maintain proper body posture, especially waist
        # Joint order: 0-5: left leg, 6-11: right leg, 12: waist_yaw, 13: waist_roll, 14: waist_pitch, 15-21: left arm, 22-28: right arm
        "pose": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:pose",
            weight=-2.0,  # Increased weight to enforce better posture
            params={
                "pose_weights": [
                    # Left leg (0-5)
                    0.01, 1.0, 5.0, 0.01, 5.0, 5.0,
                    # Right leg (6-11)
                    0.01, 1.0, 5.0, 0.01, 5.0, 5.0,
                    # Waist (12-14) - critical for preventing backward leaning
                    0.01,  # waist_yaw: allow rotation
                    1.0,   # waist_roll: moderate constraint
                    20.0,  # waist_pitch: STRONG constraint to prevent backward leaning
                    # Left arm (15-21) - minimal constraint as requested
                    0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01,
                    # Right arm (22-28) - minimal constraint as requested
                    0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01,
                ],
            },
            tags=["penalty_curriculum"],
        ),
    },
)

__all__ = ["g1_29dof_loco", "g1_29dof_loco_fast_sac", "g1_29dof_loco_robust", "g1_29dof_loco_robust_refined"]
