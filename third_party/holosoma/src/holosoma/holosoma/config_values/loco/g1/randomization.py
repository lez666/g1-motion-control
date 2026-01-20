"""Locomotion randomization presets for the G1 robot."""

from holosoma.config_types.randomization import RandomizationManagerCfg, RandomizationTermCfg

g1_29dof_randomization = RandomizationManagerCfg(
    setup_terms={
        "push_randomizer_state": RandomizationTermCfg(
            func="holosoma.managers.randomization.terms.locomotion:PushRandomizerState",
            params={
                "push_interval_s": [5, 10],
                "max_push_vel": [1.0, 1.0],
                "enabled": True,
            },
        ),
        "setup_action_delay_buffers": RandomizationTermCfg(
            func="holosoma.managers.randomization.terms.locomotion:setup_action_delay_buffers",
            params={
                "ctrl_delay_step_range": [0, 1],
                "enabled": True,
            },
        ),
        "setup_torque_rfi": RandomizationTermCfg(
            func="holosoma.managers.randomization.terms.locomotion:setup_torque_rfi",
            params={
                "enabled": False,
                "rfi_lim": 0.1,
            },
        ),
        "setup_dof_pos_bias": RandomizationTermCfg(
            func="holosoma.managers.randomization.terms.locomotion:setup_dof_pos_bias",
            params={
                "dof_pos_bias_range": [-0.01, 0.01],
                "enabled": False,
            },
        ),
        "actuator_randomizer_state": RandomizationTermCfg(
            func="holosoma.managers.randomization.terms.locomotion:ActuatorRandomizerState",
            params={
                "kp_range": [0.9, 1.1],
                "kd_range": [0.9, 1.1],
                "rfi_lim_range": [0.5, 1.5],
                "enable_pd_gain": True,
                "enable_rfi_lim": False,
            },
        ),
        "mass_randomizer": RandomizationTermCfg(
            func="holosoma.managers.randomization.terms.locomotion:randomize_mass_startup",
            params={
                "enable_link_mass": True,
                "link_mass_range": [0.9, 1.2],
                "enable_base_mass": True,
                "added_mass_range": [-1.0, 3.0],
            },
        ),
        "randomize_friction_startup": RandomizationTermCfg(
            func="holosoma.managers.randomization.terms.locomotion:randomize_friction_startup",
            params={
                "friction_range": [0.5, 1.25],
                "enabled": True,
            },
        ),
        "randomize_base_com_startup": RandomizationTermCfg(
            func="holosoma.managers.randomization.terms.locomotion:randomize_base_com_startup",
            params={
                "base_com_range": {"x": [-0.05, 0.05], "y": [-0.05, 0.05], "z": [-0.05, 0.05]},
                "enabled": True,
            },
        ),
    },
    reset_terms={
        "push_randomizer_state": RandomizationTermCfg(
            func="holosoma.managers.randomization.terms.locomotion:PushRandomizerState"
        ),
        "actuator_randomizer_state": RandomizationTermCfg(
            func="holosoma.managers.randomization.terms.locomotion:ActuatorRandomizerState"
        ),
        "randomize_push_schedule": RandomizationTermCfg(
            func="holosoma.managers.randomization.terms.locomotion:randomize_push_schedule",
        ),
        "randomize_action_delay": RandomizationTermCfg(
            func="holosoma.managers.randomization.terms.locomotion:randomize_action_delay",
        ),
        "randomize_dof_state": RandomizationTermCfg(
            func="holosoma.managers.randomization.terms.locomotion:randomize_dof_state",
            params={
                "joint_pos_scale_range": [0.5, 1.5],
                "joint_pos_bias_range": [0.0, 0.0],
                "joint_vel_range": [0.0, 0.0],
                "randomize_dof_pos_bias": False,
            },
        ),
        "configure_torque_rfi": RandomizationTermCfg(
            func="holosoma.managers.randomization.terms.locomotion:configure_torque_rfi",
        ),
    },
    step_terms={
        "push_randomizer_state": RandomizationTermCfg(
            func="holosoma.managers.randomization.terms.locomotion:PushRandomizerState"
        ),
        "apply_pushes": RandomizationTermCfg(
            func="holosoma.managers.randomization.terms.locomotion:apply_pushes",
        ),
    },
)

g1_29dof_randomization_robust = RandomizationManagerCfg(
    setup_terms={
        # Physical material randomization (friction, restitution) - matches prompt file
        "randomize_robot_rigid_body_material": RandomizationTermCfg(
            func="holosoma.managers.randomization.terms.locomotion:randomize_robot_rigid_body_material_startup",
            params={
                "static_friction_range": (0.4, 1.0),
                "dynamic_friction_range": (0.4, 1.0),
                "restitution_range": (0.0, 0.0),
                "num_buckets": 64,
                "enabled": True,
            },
        ),
        # Mass randomization (body mass scaling + torso additional mass)
        "mass_randomizer": RandomizationTermCfg(
            func="holosoma.managers.randomization.terms.locomotion:randomize_mass_startup",
            params={
                "enable_link_mass": True,
                "link_mass_range": [0.9, 1.1],  # Body mass scaling (matches prompt file)
                "enable_base_mass": True,
                "added_mass_range": [-1.0, 1.0],  # Torso additional mass (kg) - matches prompt file
                "enabled": True,
            },
        ),
        # Actuator randomization (PD gains, armature/joint friction)
        "actuator_randomizer_state": RandomizationTermCfg(
            func="holosoma.managers.randomization.terms.locomotion:ActuatorRandomizerState",
            params={
                "kp_range": [0.9, 1.1],  # PD gain scaling
                "kd_range": [0.9, 1.1],  # PD damping scaling
                "rfi_lim_range": [1.0, 1.05],  # Armature scaling (joint friction/inertia) - matches prompt file
                "enable_pd_gain": True,
                "enable_rfi_lim": True,  # Enable armature randomization
            },
        ),
        # Joint friction randomization setup
        "setup_torque_rfi": RandomizationTermCfg(
            func="holosoma.managers.randomization.terms.locomotion:setup_torque_rfi",
            params={
                "enabled": True,
                "rfi_lim": 0.1,
            },
        ),
        # Push randomizer (external disturbances) - moderate intensity from start
        "push_randomizer_state": RandomizationTermCfg(
            func="holosoma.managers.randomization.terms.locomotion:PushRandomizerState",
            params={
                "push_interval_s": [2.0, 4.0],  # Interval between pushes (seconds) - matches prompt file
                "max_push_vel": [0.5, 0.5],  # Max push velocity in x, y directions (m/s) - moderate intensity
                "enabled": True,  # Enabled from start (adaptive curriculum will handle difficulty)
            },
        ),
        # Action delay buffers
        "setup_action_delay_buffers": RandomizationTermCfg(
            func="holosoma.managers.randomization.terms.locomotion:setup_action_delay_buffers",
            params={
                "ctrl_delay_step_range": [0, 1],
                "enabled": True,
            },
        ),
        # Base COM randomization
        "randomize_base_com_startup": RandomizationTermCfg(
            func="holosoma.managers.randomization.terms.locomotion:randomize_base_com_startup",
            params={
                "base_com_range": {"x": [-0.05, 0.05], "y": [-0.05, 0.05], "z": [-0.05, 0.05]},
                "enabled": True,
            },
        ),
    },
    reset_terms={
        # Push randomizer reset
        "push_randomizer_state": RandomizationTermCfg(
            func="holosoma.managers.randomization.terms.locomotion:PushRandomizerState"
        ),
        # Actuator randomizer reset
        "actuator_randomizer_state": RandomizationTermCfg(
            func="holosoma.managers.randomization.terms.locomotion:ActuatorRandomizerState"
        ),
        # Push schedule randomization
        "randomize_push_schedule": RandomizationTermCfg(
            func="holosoma.managers.randomization.terms.locomotion:randomize_push_schedule",
        ),
        # Action delay randomization
        "randomize_action_delay": RandomizationTermCfg(
            func="holosoma.managers.randomization.terms.locomotion:randomize_action_delay",
        ),
        # DOF state randomization (joint positions and velocities)
        # Note: Root state randomization (position, velocity) is handled in locomotion_manager._reset_root_states
        # with ranges: pos x,y: (-0.5, 0.5), yaw: (-π, π), vel: (-0.5, 0.5) - matches prompt file
        "randomize_dof_state": RandomizationTermCfg(
            func="holosoma.managers.randomization.terms.locomotion:randomize_dof_state",
            params={
                "joint_pos_scale_range": [0.8, 1.2],  # Joint position scaling (0.8-1.2x default) - reduced for better stability
                "joint_pos_bias_range": [0.0, 0.0],
                "joint_vel_range": [0.0, 0.0],  # Joint velocities start at zero
                "randomize_dof_pos_bias": False,
            },
        ),
        # Torque RFI configuration
        "configure_torque_rfi": RandomizationTermCfg(
            func="holosoma.managers.randomization.terms.locomotion:configure_torque_rfi",
        ),
    },
    step_terms={
        # Push randomizer step
        "push_randomizer_state": RandomizationTermCfg(
            func="holosoma.managers.randomization.terms.locomotion:PushRandomizerState"
        ),
        # Apply pushes
        "apply_pushes": RandomizationTermCfg(
            func="holosoma.managers.randomization.terms.locomotion:apply_pushes",
        ),
    },
)

__all__ = ["g1_29dof_randomization", "g1_29dof_randomization_robust"]
