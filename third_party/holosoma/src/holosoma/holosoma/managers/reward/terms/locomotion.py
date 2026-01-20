"""Reward terms for locomotion tasks.

These terms are migrated from LeggedRobotBase._reward_* methods to be
compatible with the reward manager system.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from holosoma.managers.observation.terms.locomotion import (
    base_forward_vector,
    get_base_ang_vel,
    get_base_lin_vel,
    get_projected_gravity,
    gravity_vector,
)
from holosoma.utils.rotations import (
    quat_apply,
    quat_rotate_batched,
    quat_rotate_inverse,
)
from holosoma.utils.safe_torch_import import torch

if TYPE_CHECKING:
    from holosoma.envs.locomotion.locomotion_manager import LeggedRobotLocomotionManager


def _expected_foot_height(phi: torch.Tensor, swing_height: float) -> torch.Tensor:
    """Expected foot height from gait phase using a cubic Bézier profile."""

    def cubic_bezier_interpolation(y_start: torch.Tensor, y_end: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        y_diff = y_end - y_start
        bezier = x**3 + 3 * (x**2 * (1 - x))
        return y_start + y_diff * bezier

    x = (phi + torch.pi) / (2 * torch.pi)
    stance = cubic_bezier_interpolation(torch.zeros_like(x), torch.full_like(x, swing_height), 2 * x)
    swing = cubic_bezier_interpolation(torch.full_like(x, swing_height), torch.zeros_like(x), 2 * x - 1)
    return torch.where(x <= 0.5, stance, swing)


# ================================================================================================
# Termination Rewards
# ================================================================================================


def termination(env: LeggedRobotLocomotionManager) -> torch.Tensor:
    """Terminal reward/penalty for early termination (excluding timeouts).

    Args:
        env: The environment instance

    Returns:
        Reward tensor [num_envs]
    """
    return (env.reset_buf * ~env.time_out_buf).float()


# ================================================================================================
# Penalty Rewards
# ================================================================================================


def penalty_action_rate(env: LeggedRobotLocomotionManager) -> torch.Tensor:
    """Penalize changes in actions between steps.

    Args:
        env: The environment instance

    Returns:
        Reward tensor [num_envs]
    """
    actions = env.action_manager.action
    prev_actions = env.action_manager.prev_action
    return torch.sum(torch.square(prev_actions - actions), dim=1)


def penalty_orientation(env: LeggedRobotLocomotionManager) -> torch.Tensor:
    """Penalize non-flat base orientation.

    Args:
        env: The environment instance

    Returns:
        Reward tensor [num_envs]
    """
    projected = get_projected_gravity(env)
    return torch.sum(torch.square(projected[:, :2]), dim=1)


def penalty_feet_ori(env: LeggedRobotLocomotionManager) -> torch.Tensor:
    """Penalize feet orientation deviation from flat.

    Args:
        env: The environment instance

    Returns:
        Reward tensor [num_envs]
    """
    left_quat = env.simulator._rigid_body_rot[:, env.feet_indices[0]]
    gravity = gravity_vector(env)
    left_gravity = quat_rotate_inverse(left_quat, gravity, w_last=True)
    right_quat = env.simulator._rigid_body_rot[:, env.feet_indices[1]]
    right_gravity = quat_rotate_inverse(right_quat, gravity, w_last=True)
    return (
        torch.sum(torch.square(left_gravity[:, :2]), dim=1) ** 0.5
        + torch.sum(torch.square(right_gravity[:, :2]), dim=1) ** 0.5
    )


# ================================================================================================
# Limit Rewards
# ================================================================================================


def limits_dof_pos(env: LeggedRobotLocomotionManager, soft_dof_pos_limit: float = 0.95) -> torch.Tensor:
    """Penalize joint positions too close to limits.

    Args:
        env: The environment instance
        soft_dof_pos_limit: Soft limit as fraction of hard limit

    Returns:
        Reward tensor [num_envs]
    """
    # Use soft limits as fraction of hard limits
    m = (env.simulator.hard_dof_pos_limits[:, 0] + env.simulator.hard_dof_pos_limits[:, 1]) / 2  # type: ignore[attr-defined]
    r = env.simulator.hard_dof_pos_limits[:, 1] - env.simulator.hard_dof_pos_limits[:, 0]  # type: ignore[attr-defined]
    lower_soft_limit = m - 0.5 * r * soft_dof_pos_limit
    upper_soft_limit = m + 0.5 * r * soft_dof_pos_limit

    out_of_limits = -(env.simulator.dof_pos - lower_soft_limit).clip(max=0.0)  # lower limit
    out_of_limits += (env.simulator.dof_pos - upper_soft_limit).clip(min=0.0)
    return torch.sum(out_of_limits, dim=1)


# ================================================================================================
# Tracking and Task Rewards
# ================================================================================================


def tracking_lin_vel(env, tracking_sigma: float = 0.25) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes).

    Uses exponential reward: exp(-error / sigma)

    Args:
        env: The environment instance
        tracking_sigma: Sigma for exponential reward scaling

    Returns:
        Reward tensor [num_envs]
    """
    commands = env.command_manager.commands
    lin_vel_error = torch.sum(torch.square(commands[:, :2] - get_base_lin_vel(env)[:, :2]), dim=1)
    return torch.exp(-lin_vel_error / tracking_sigma)


def tracking_ang_vel(env, tracking_sigma: float = 0.25) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw).

    Uses exponential reward: exp(-error / sigma)

    Args:
        env: The environment instance
        tracking_sigma: Sigma for exponential reward scaling

    Returns:
        Reward tensor [num_envs]
    """
    commands = env.command_manager.commands
    ang_vel = get_base_ang_vel(env)
    ang_vel_error = torch.square(commands[:, 2] - ang_vel[:, 2])
    return torch.exp(-ang_vel_error / tracking_sigma)


def penalty_ang_vel_xy(env) -> torch.Tensor:
    """Penalize xy axes base angular velocity.

    Args:
        env: The environment instance

    Returns:
        Reward tensor [num_envs]
    """
    ang_vel = get_base_ang_vel(env)
    return torch.sum(torch.square(ang_vel[:, :2]), dim=1)


def penalty_close_feet_xy(env, close_feet_threshold: float = 0.05) -> torch.Tensor:
    """Penalize when feet are too close together in xy plane.

    Args:
        env: The environment instance
        close_feet_threshold: Minimum distance threshold between feet

    Returns:
        Reward tensor [num_envs]
    """
    left_foot_xy = env.simulator._rigid_body_pos[:, env.feet_indices[0], :2]
    right_foot_xy = env.simulator._rigid_body_pos[:, env.feet_indices[1], :2]

    # Get base orientation
    base_forward = quat_apply(env.base_quat, base_forward_vector(env), w_last=True)
    base_yaw = torch.atan2(base_forward[:, 1], base_forward[:, 0])

    # Calculate perpendicular distance in base-local coordinates
    feet_distance = torch.abs(
        torch.cos(base_yaw) * (left_foot_xy[:, 1] - right_foot_xy[:, 1])
        - torch.sin(base_yaw) * (left_foot_xy[:, 0] - right_foot_xy[:, 0])
    )

    # Return penalty when feet are too close
    return (feet_distance < close_feet_threshold).float()


def base_height(
    env, desired_base_height: float = 0.89, zero_vel_penalty_scale: float = 1.0, stance_penalty_scale: float = 1.0
) -> torch.Tensor:
    """Penalize base height away from target.

    Args:
        env: The environment instance
        desired_base_height: Target base height
        zero_vel_penalty_scale: Multiplier for base height penalty when robot has zero velocity commands
        stance_penalty_scale: Multiplier for base height penalty when robot is in stance mode

    Returns:
        Reward tensor [num_envs]
    """
    base_height_penalty = torch.square(
        env.terrain_manager.get_state("locomotion_terrain").base_heights - desired_base_height
    )

    # Apply stronger penalty for zero velocity commands if configured
    if zero_vel_penalty_scale != 1.0:
        commands = env.command_manager.commands
        zero_vel_mask = torch.norm(commands[:, :2], dim=1) < 0.1
        base_height_penalty = torch.where(
            zero_vel_mask, base_height_penalty * zero_vel_penalty_scale, base_height_penalty
        )

    # Apply stronger penalty for stance mode if configured (used in decoupled locomotion)
    if stance_penalty_scale != 1.0 and hasattr(env, "stance_mask"):
        base_height_penalty = torch.where(
            env.stance_mask, base_height_penalty * stance_penalty_scale, base_height_penalty
        )

    return base_height_penalty


def feet_phase(env, swing_height: float = 0.08, tracking_sigma: float = 0.25) -> torch.Tensor:
    """Reward for tracking desired foot height based on gait phase.

    Based on MuJoCo Playground's implementation.

    Args:
        env: The environment instance
        swing_height: Maximum height during swing phase
        tracking_sigma: Sigma for exponential reward scaling

    Returns:
        Reward tensor [num_envs]
    """
    # Get foot heights (relative to terrain)
    foot_z_left = env.terrain_manager.get_state("locomotion_terrain").feet_heights[:, 0]
    foot_z_right = env.terrain_manager.get_state("locomotion_terrain").feet_heights[:, 1]

    # Calculate expected foot heights based on phase
    gait_state = env.command_manager.get_state("locomotion_gait")
    rz_left = _expected_foot_height(gait_state.phase[:, 0], swing_height)
    rz_right = _expected_foot_height(gait_state.phase[:, 1], swing_height)

    # Calculate height tracking errors
    error_left = torch.square(foot_z_left - rz_left)
    error_right = torch.square(foot_z_right - rz_right)

    # Combine errors and apply exponential reward
    total_error = error_left + error_right

    return torch.exp(-total_error / tracking_sigma)


def pose(
    env,
    pose_weights: list[float],
) -> torch.Tensor:
    """Reward for maintaining default pose.

    Penalizes deviation from default joint positions with weighted importance.

    Args:
        env: The environment instance
        pose_weights: List of weights for each DOF (must match num_dof)

    Returns:
        Reward tensor [num_envs]
    """
    # Get current joint positions
    qpos = env.simulator.dof_pos

    # Convert pose_weights to tensor
    weights = torch.tensor(pose_weights, device=env.device, dtype=torch.float32)

    # Calculate squared deviation from default pose
    # Use env.default_dof_pos which is already set up from robot config
    pose_error = torch.square(qpos - env.default_dof_pos)

    # Weight and sum the errors
    weighted_error = pose_error * weights.unsqueeze(0)

    return torch.sum(weighted_error, dim=1)


def penalty_stumble(env) -> torch.Tensor:
    """Penalize feet hitting vertical surfaces.

    Args:
        env: The environment instance

    Returns:
        Reward tensor [num_envs]
    """
    return torch.any(
        torch.norm(env.simulator.contact_forces[:, env.feet_indices, :2], dim=2)
        > 4 * torch.abs(env.simulator.contact_forces[:, env.feet_indices, 2]),
        dim=1,
    )


def penalty_foothold(env, foothold_epsilon: float = 0.01) -> torch.Tensor:
    """Sampling-based foothold penalty.

    For each foot in contact, sample a grid of points on the sole, transform to world,
    read terrain height at those XY, compute depth d_ij = z_sample - terrain_z, and count
    samples with d_ij < epsilon. Sum over both feet.

    Args:
        env: The environment instance
        foothold_epsilon: Threshold for foothold depth penalty

    Returns:
        Reward tensor [num_envs]
    """
    # Contact mask per foot
    contact = env.simulator.contact_forces[:, env.feet_indices, 2] > 1.0  # [E,2]
    if not (contact.any()):
        return torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)

    # Accumulator
    penalty = torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)

    for foot_idx_local in range(2):
        # Skip if no env has contact on this foot to save work
        if not contact[:, foot_idx_local].any():
            continue
        rb_idx = env.feet_indices[foot_idx_local]
        foot_pos_w = env.simulator._rigid_body_pos[:, rb_idx, :]  # [E,3]
        foot_quat_w = env.simulator._rigid_body_rot[:, rb_idx, :]  # [E,4]

        # Use precomputed sample points in the foot frame
        pts_local = env.foot_samples_local[foot_idx_local].unsqueeze(0).repeat(env.num_envs, 1, 1)

        # Rotate to world and translate
        pts_world = quat_rotate_batched(foot_quat_w, pts_local) + foot_pos_w.unsqueeze(1)

        # Query terrain height at those XY positions
        terrain_h = env._get_terrain_heights_at_points_world(pts_world)

        # Depth: world z minus terrain height
        depth = pts_world[:, :, 2] - terrain_h  # [E,S]

        # Indicator for d_ij > epsilon, only for envs with this foot in contact
        bad = (depth > foothold_epsilon).float()
        bad *= contact[:, foot_idx_local].unsqueeze(1).float()

        penalty += torch.sum(bad, dim=1)

    return penalty / env.num_foot_samples


def alive(env) -> torch.Tensor:
    """Reward for staying alive.

    Args:
        env: The environment instance

    Returns:
        Reward tensor [num_envs]
    """
    return torch.ones(env.num_envs, dtype=torch.float, device=env.device)


# ================================================================================================
# Additional Reward Functions for Robust Locomotion
# ================================================================================================


def flat_orientation_l2(env: LeggedRobotLocomotionManager) -> torch.Tensor:
    """Reward for flat base orientation (L2 norm of projected gravity xy components).

    Args:
        env: The environment instance

    Returns:
        Reward tensor [num_envs] - negative value (penalty for non-flat orientation)
    """
    projected = get_projected_gravity(env)
    return -torch.sum(torch.square(projected[:, :2]), dim=1)


def feet_air_time_positive_biped(
    env: LeggedRobotLocomotionManager, threshold: float = 0.2
) -> torch.Tensor:
    """Reward for positive foot air time during swing phase for bipedal robots.

    Only rewards when robot is moving (non-zero velocity command).

    Args:
        env: The environment instance
        threshold: Minimum contact force threshold to consider foot in contact

    Returns:
        Reward tensor [num_envs]
    """
    # Get contact forces for feet
    contact_forces = env.simulator.contact_forces[:, env.feet_indices, 2]  # [num_envs, 2]
    
    # Check if feet are in air (contact force below threshold)
    feet_in_air = contact_forces < threshold  # [num_envs, 2]
    
    # Only reward when robot is moving (check velocity commands)
    commands = env.command_manager.commands
    is_moving = torch.norm(commands[:, :2], dim=1) > 0.01  # [num_envs]
    
    # Reward positive air time (feet should be in air during swing)
    # This is a simplified version - in practice, you'd track air time over multiple steps
    air_time_reward = torch.sum(feet_in_air.float(), dim=1)  # [num_envs]
    
    # Only apply reward when moving
    return torch.where(is_moving, air_time_reward, torch.zeros_like(air_time_reward))


def both_feet_air(env: LeggedRobotLocomotionManager, threshold: float = 0.2) -> torch.Tensor:
    """Penalty when both feet are in the air simultaneously (should not happen for stable walking).

    Args:
        env: The environment instance
        threshold: Minimum contact force threshold to consider foot in contact

    Returns:
        Penalty tensor [num_envs] - 1.0 when both feet are in air, 0.0 otherwise
    """
    # Get contact forces for feet
    contact_forces = env.simulator.contact_forces[:, env.feet_indices, 2]  # [num_envs, 2]
    
    # Check if both feet are in air
    both_in_air = torch.all(contact_forces < threshold, dim=1)  # [num_envs]
    
    return both_in_air.float()


def feet_slide(env: LeggedRobotLocomotionManager, threshold: float = 0.2) -> torch.Tensor:
    """Penalty for feet sliding when in contact with ground.

    Args:
        env: The environment instance
        threshold: Minimum contact force threshold to consider foot in contact

    Returns:
        Penalty tensor [num_envs] - sum of lateral contact forces for feet in contact
    """
    # Get contact forces for feet [num_envs, 2, 3]
    contact_forces = env.simulator.contact_forces[:, env.feet_indices, :]
    
    # Check which feet are in contact (z component > threshold)
    in_contact = contact_forces[:, :, 2] > threshold  # [num_envs, 2]
    
    # Calculate lateral (xy) contact forces
    lateral_forces = torch.norm(contact_forces[:, :, :2], dim=2)  # [num_envs, 2]
    
    # Only penalize when foot is in contact
    slide_penalty = torch.where(in_contact, lateral_forces, torch.zeros_like(lateral_forces))
    
    # Sum over both feet
    return torch.sum(slide_penalty, dim=1)


def joint_torques_l2(
    env: LeggedRobotLocomotionManager, joint_names: list[str] | None = None
) -> torch.Tensor:
    """Penalty for joint torques (L2 norm).

    Args:
        env: The environment instance
        joint_names: Optional list of joint names to include. If None, includes all joints.

    Returns:
        Penalty tensor [num_envs] - L2 norm of joint torques
    """
    # Get torques from action manager
    try:
        joint_control_term = env.action_manager.get_term("joint_control")
        torques = joint_control_term.torques  # [num_envs, num_dof]
    except (AttributeError, KeyError):
        # Fallback: try to get from simulator if available
        if hasattr(env.simulator, "dof_forces"):
            torques = env.simulator.dof_forces  # [num_envs, num_dof]
        else:
            # If torques not available, return zeros
            return torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)
    
    # Filter by joint names if specified
    if joint_names is not None:
        joint_indices = []
        for name in joint_names:
            # Use regex matching similar to Isaac Lab
            import re
            pattern = re.compile(name)
            for i, dof_name in enumerate(env.dof_names):
                if pattern.match(dof_name):
                    joint_indices.append(i)
        if joint_indices:
            torques = torques[:, joint_indices]
        else:
            return torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)
    
    # Calculate L2 norm
    return torch.sum(torch.square(torques), dim=1)


def joint_acc_l2(
    env: LeggedRobotLocomotionManager, joint_names: list[str] | None = None
) -> torch.Tensor:
    """Penalty for joint accelerations (L2 norm).

    Args:
        env: The environment instance
        joint_names: Optional list of joint names to include. If None, includes all joints.

    Returns:
        Penalty tensor [num_envs] - L2 norm of joint accelerations
    """
    # Get accelerations from simulator
    if not hasattr(env.simulator, "dof_acc"):
        return torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)
    
    acc = env.simulator.dof_acc  # [num_envs, num_dof]
    
    # Filter by joint names if specified
    if joint_names is not None:
        joint_indices = []
        for name in joint_names:
            # Use regex matching similar to Isaac Lab
            import re
            pattern = re.compile(name)
            for i, dof_name in enumerate(env.dof_names):
                if pattern.match(dof_name):
                    joint_indices.append(i)
        if joint_indices:
            acc = acc[:, joint_indices]
        else:
            return torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)
    
    # Calculate L2 norm
    return torch.sum(torch.square(acc), dim=1)


def joint_pos_limits(
    env: LeggedRobotLocomotionManager, joint_names: list[str] | None = None
) -> torch.Tensor:
    """Penalty for joint positions exceeding limits.

    Args:
        env: The environment instance
        joint_names: Optional list of joint names to include. If None, includes all joints.

    Returns:
        Penalty tensor [num_envs] - sum of violations
    """
    dof_pos = env.simulator.dof_pos  # [num_envs, num_dof]
    hard_limits = env.simulator.hard_dof_pos_limits  # [num_dof, 2]
    
    # Filter by joint names if specified
    if joint_names is not None:
        joint_indices = []
        for name in joint_names:
            # Use regex matching similar to Isaac Lab
            import re
            pattern = re.compile(name)
            for i, dof_name in enumerate(env.dof_names):
                if pattern.match(dof_name):
                    joint_indices.append(i)
        if joint_indices:
            dof_pos = dof_pos[:, joint_indices]
            hard_limits = hard_limits[joint_indices, :]
        else:
            return torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)
    
    # Check violations
    lower_violation = -(dof_pos - hard_limits[:, 0]).clip(max=0.0)
    upper_violation = (dof_pos - hard_limits[:, 1]).clip(min=0.0)
    
    return torch.sum(lower_violation + upper_violation, dim=1)


def joint_deviation_l1(env: LeggedRobotLocomotionManager) -> torch.Tensor:
    """Penalty for joint position deviation from default (L1 norm).

    Args:
        env: The environment instance

    Returns:
        Penalty tensor [num_envs] - L1 norm of joint deviations
    """
    dof_pos = env.simulator.dof_pos  # [num_envs, num_dof]
    default_pos = env.default_dof_pos  # [num_envs, num_dof]
    
    deviation = torch.abs(dof_pos - default_pos)
    return torch.sum(deviation, dim=1)
