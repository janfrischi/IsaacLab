# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import subtract_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# Define the observation functions for the lift task

# Return object position in the robot's root frame x, y, z
def object_position_in_robot_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """The position of the object in the robot's root frame."""
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    object_pos_w = object.data.root_pos_w[:, :3]
    object_pos_b, _ = subtract_frame_transforms(
        robot.data.root_state_w[:, :3], # robot position in the world frame
        robot.data.root_state_w[:, 3:7], # robot orientation in the world frame
        object_pos_w # object position in the world frame
    )
    return object_pos_b

# Return object orientation in the robot's root frame as quaternion x, y, z, w
def object_orientation_in_robot_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """The orientation of the object as a quaternion in the robot's frame."""
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    object_quat_w = object.data.root_quat_w
    _, object_quat_b = subtract_frame_transforms(
        robot.data.root_state_w[:, :3], # robot position in the world frame
        robot.data.root_state_w[:, 3:7], # robot orientation in the world frame
        None, # object position in the world frame -> Not needed
        object_quat_w
    )
    return object_quat_b