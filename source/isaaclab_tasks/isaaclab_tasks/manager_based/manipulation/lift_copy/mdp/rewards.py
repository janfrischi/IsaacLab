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

# Reward agent for lifting the object above the minimal height. -> Binary reward function
def object_is_lifted(
    env: ManagerBasedRLEnv, minimal_height: float, object_cfg: SceneEntityCfg = SceneEntityCfg("object")
) -> torch.Tensor:
    """Reward the agent for lifting the object above the minimal height."""
    object: RigidObject = env.scene[object_cfg.name]
    # Return a binary tensor [:,2] extracts the z coordinate of the object
    return torch.where(object.data.root_pos_w[:, 2] > minimal_height, 1.0, 0.0)

# Reward agent for moving the end-effector closer to the object. Use tanh-kernel to reward the agent for reaching the object.
# The closer the ee the higher the reward.-> Encourage the agent to move the end-effector closer to the object.
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
    # Distance of the end-effector to the object: (num_envs,) L2 norm
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
    
    # Extract robot & object states
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)

    # Compute desired position in the base frame xyz and the quaterion
    des_pos_b, des_quat_b = command[:, :3], command[:, 3:7]
    # Convert the goal position and orientation from local frame to base frame
    des_pos_w, des_quat_w = combine_frame_transforms(
        robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], 
        des_pos_b, des_quat_b
    )

    # Compute position distance -> Encourage agent to reduce this distance over time
    distance = torch.norm(des_pos_w - object.data.root_pos_w[:, :3], dim=1)

    # Compute orientation difference (corrected quaternion distance)
    object_quat_w = object.data.root_quat_w
    quat_diff = torch.abs(torch.sum(des_quat_w * object_quat_w, dim=1))
    # 2*acos(...) converts it into a full angle distance in radiands
    quat_diff = 2 * torch.acos(torch.clamp(quat_diff, -1, 1))

    # Compute tanh-based penalties
    position_penalty = 1 - torch.tanh(distance / std)
    # Normalize the orientation penalty to be in [0, 1]
    orientation_penalty = 1 - torch.tanh(0.5 * quat_diff / 3.14)

    # Final reward calculation (only when object is lifted above minimal height) , too heigh weight for orientation lead to oscillations
    return (object.data.root_pos_w[:, 2] > minimal_height) * (position_penalty + 1.5*orientation_penalty)




