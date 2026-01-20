"""Locomotion command presets for the G1 robot."""

from holosoma.config_types.command import CommandManagerCfg, CommandTermCfg

g1_29dof_command = CommandManagerCfg(
    params={
        "locomotion_command_resampling_time": 10.0,
    },
    setup_terms={
        "locomotion_gait": CommandTermCfg(
            func="holosoma.managers.command.terms.locomotion:LocomotionGait",
            params={
                "gait_period": 1.0,
                "gait_period_randomization_width": 0.2,
            },
        ),
        "locomotion_command": CommandTermCfg(
            func="holosoma.managers.command.terms.locomotion:LocomotionCommand",
            params={
                "command_ranges": {
                    "lin_vel_x": [-1.0, 1.0],
                    "lin_vel_y": [-1.0, 1.0],
                    "ang_vel_yaw": [-1.0, 1.0],
                    "heading": [-3.14, 3.14],
                },
                "stand_prob": 0.2,
            },
        ),
    },
    reset_terms={
        "locomotion_gait": CommandTermCfg(func="holosoma.managers.command.terms.locomotion:LocomotionGait"),
        "locomotion_command": CommandTermCfg(func="holosoma.managers.command.terms.locomotion:LocomotionCommand"),
    },
    step_terms={
        "locomotion_gait": CommandTermCfg(func="holosoma.managers.command.terms.locomotion:LocomotionGait"),
        "locomotion_command": CommandTermCfg(func="holosoma.managers.command.terms.locomotion:LocomotionCommand"),
    },
)

g1_29dof_command_robust = CommandManagerCfg(
    params={
        # Random resampling time between 3.0 and 8.0 seconds for training diversity
        "locomotion_command_resampling_time": 5.5,  # Mean of (3.0, 8.0) - actual randomization handled in command term
    },
    setup_terms={
        "locomotion_gait": CommandTermCfg(
            func="holosoma.managers.command.terms.locomotion:LocomotionGait",
            params={
                "gait_period": 1.0,
                "gait_period_randomization_width": 0.2,
            },
        ),
        "locomotion_command": CommandTermCfg(
            func="holosoma.managers.command.terms.locomotion:LocomotionCommand",
            params={
                "command_ranges": {
                    # Extended velocity ranges to match keyboard control interface
                    # lin_vel_x: forward/backward (corresponds to keyboard up/down arrows)
                    "lin_vel_x": [-2.0, 2.0],  # Increased from [-1.0, 1.0] for keyboard control compatibility
                    # lin_vel_y: lateral movement (corresponds to keyboard left/right arrows)
                    "lin_vel_y": [-1.0, 1.0],  # Matches keyboard control max_lat_vel
                    # ang_vel_yaw: rotation (corresponds to keyboard Z/C keys)
                    "ang_vel_yaw": [-1.0, 1.0],  # Matches keyboard control max_ang_vel
                    "heading": [-3.14, 3.14],
                },
                "stand_prob": 0.5,  # Increased from 0.2 to 0.5 - more time learning to stand
            },
        ),
    },
    reset_terms={
        "locomotion_gait": CommandTermCfg(func="holosoma.managers.command.terms.locomotion:LocomotionGait"),
        "locomotion_command": CommandTermCfg(func="holosoma.managers.command.terms.locomotion:LocomotionCommand"),
    },
    step_terms={
        "locomotion_gait": CommandTermCfg(func="holosoma.managers.command.terms.locomotion:LocomotionGait"),
        "locomotion_command": CommandTermCfg(func="holosoma.managers.command.terms.locomotion:LocomotionCommand"),
    },
)

__all__ = ["g1_29dof_command", "g1_29dof_command_robust"]
