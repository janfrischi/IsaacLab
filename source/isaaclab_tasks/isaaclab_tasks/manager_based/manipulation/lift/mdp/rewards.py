# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import numpy as np  
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer
from isaaclab.utils.math import combine_frame_transforms, matrix_from_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def object_is_lifted(
    env: ManagerBasedRLEnv, minimal_height: float, object_cfg: SceneEntityCfg = SceneEntityCfg("object")
) -> torch.Tensor:
    """Reward the agent for lifting the object above the minimal height."""
    object: RigidObject = env.scene[object_cfg.name]
    return torch.where(object.data.root_pos_w[:, 2] > minimal_height, 1.0, 0.0)

# Reward agent for moving the end-effector closer to the object. Use tanh-kernel to reward the agent for reaching the object.
# The closer the ee the higher the reward.
def object_ee_distance(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward the agent for reaching the object using tanh-kernel."""
    # extract the used quantities (to enable type-hinting)
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    # Target object position: (num_envs, 3)
    cube_pos_w = object.data.root_pos_w
    # End-effector position: (num_envs, 3)
    ee_w = ee_frame.data.target_pos_w[..., 0, :]
    # Distance of the end-effector to the object: (num_envs,)
    object_ee_distance = torch.norm(cube_pos_w - ee_w, dim=1)

    return 1 - torch.tanh(object_ee_distance / std)

# Reward agent for moving the object closer to the goal. Use tanh-kernel to reward the agent for reaching the goal.
def object_goal_distance(
    env: ManagerBasedRLEnv,
    std: float,
    minimal_height: float,
    command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward the agent for tracking the goal pose using tanh-kernel."""
    
    # Extract robot & object states
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)

    # Compute desired world position & quaternion
    des_pos_b, des_quat_b = command[:, :3], command[:, 3:7]
    des_pos_w, des_quat_w = combine_frame_transforms(
        robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], 
        des_pos_b, des_quat_b
    )

    # Compute position distance
    distance = torch.norm(des_pos_w - object.data.root_pos_w[:, :3], dim=1)

    # Compute orientation difference (corrected quaternion distance)
    object_quat_w = object.data.root_quat_w
    quat_diff = torch.abs(torch.sum(des_quat_w * object_quat_w, dim=1))
    quat_diff = 2 * torch.acos(torch.clamp(quat_diff, -1, 1))  # Convert to angular error

    # Compute tanh-based penalties
    position_penalty = 1 - torch.tanh(distance / std)
    orientation_penalty = 1 - torch.tanh(0.5 * quat_diff / 3.14)  # Normalize by π

    # Final reward calculation (only when object is lifted)
    return (object.data.root_pos_w[:, 2] > minimal_height) * (position_penalty + orientation_penalty)


""" def gripper_alignment_with_cube(
    env: ManagerBasedRLEnv,
    std: float,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    
    # Extract scene entities
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]

    # Get gripper and object orientations (quaternions)
    gripper_quat = ee_frame.data.target_quat_w[..., 0, :]  # (num_envs, 4)
    cube_quat = object.data.root_quat_w  # (num_envs, 4)

    # Convert quaternions to rotation matrices
    gripper_rot_mat = matrix_from_quat(gripper_quat)  # (num_envs, 3, 3)
    cube_rot_mat = matrix_from_quat(cube_quat)  # (num_envs, 3, 3)

    # Extract the Z-axis (approach direction) of the gripper
    gripper_z_axis = gripper_rot_mat[:, :, 2]  # (num_envs, 3)

    # Extract the Z-axis (surface normal) of the cube
    cube_z_axis = cube_rot_mat[:, :, 2]  # (num_envs, 3)

    # Normalize vectors
    gripper_z_axis = gripper_z_axis / torch.norm(gripper_z_axis, dim=1, keepdim=True)
    cube_z_axis = cube_z_axis / torch.norm(cube_z_axis, dim=1, keepdim=True)

    # Compute alignment (dot product should be close to 1 when aligned)
    alignment_score = torch.sum(gripper_z_axis * (-cube_z_axis), dim=1)  # (num_envs,)

    # Normalize alignment score to be in [0, 1] (Shift range from [-1,1] to [0,1])
    alignment_score = (alignment_score + 1) / 2  

    # Compute reward using tanh kernel
    reward = 1 - torch.tanh((1 - alignment_score) / std)

    # Ensure reward is never exactly zero (for GUI logging)
    reward = torch.where(reward == 0.0, torch.tensor(1e-5, device=reward.device), reward)

    # Debugging print
    #print("Gripper Alignment Reward:", reward.mean().item())

    return reward """



