# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import math
import numpy as np
import random
import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, AssetBase
from isaaclab.managers import SceneEntityCfg
from isaaclab.envs import ManagerBasedRLEnv

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def set_default_joint_pose(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    default_pose: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    # Set the default pose for robots in all envs
    asset = env.scene[asset_cfg.name]
    asset.data.default_joint_pos = torch.tensor(default_pose, device=env.device).repeat(env.num_envs, 1)


def randomize_joint_by_gaussian_offset(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    mean: float,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    asset: Articulation = env.scene[asset_cfg.name]

    # Add gaussian noise to joint states
    joint_pos = asset.data.default_joint_pos[env_ids].clone()
    joint_vel = asset.data.default_joint_vel[env_ids].clone()
    joint_pos += math_utils.sample_gaussian(mean, std, joint_pos.shape, joint_pos.device)

    # Clamp joint pos to limits
    joint_pos_limits = asset.data.soft_joint_pos_limits[env_ids]
    joint_pos = joint_pos.clamp_(joint_pos_limits[..., 0], joint_pos_limits[..., 1])

    # Don't noise the gripper poses
    joint_pos[:, -2:] = asset.data.default_joint_pos[env_ids, -2:]

    # Set into the physics simulation
    asset.set_joint_position_target(joint_pos, env_ids=env_ids)
    asset.set_joint_velocity_target(joint_vel, env_ids=env_ids)
    asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)


def randomize_scene_lighting_domelight(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    intensity_range: tuple[float, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("light"),
):
    asset: AssetBase = env.scene[asset_cfg.name]
    light_prim = asset.prims[0]

    # Sample new light intensity
    new_intensity = random.uniform(intensity_range[0], intensity_range[1])

    # Set light intensity to light prim
    intensity_attr = light_prim.GetAttribute("inputs:intensity")
    intensity_attr.Set(new_intensity)


def sample_object_poses(
    num_objects: int,
    min_separation: float = 0.0,
    pose_range: dict[str, tuple[float, float]] = {},
    max_sample_tries: int = 5000,
):
    range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    pose_list = []

    for i in range(num_objects):
        for j in range(max_sample_tries):
            sample = [random.uniform(range[0], range[1]) for range in range_list]

            # Accept pose if it is the first one, or if reached max num tries
            if len(pose_list) == 0 or j == max_sample_tries - 1:
                pose_list.append(sample)
                break

            # Check if pose of object is sufficiently far away from all other objects
            separation_check = [math.dist(sample[:3], pose[:3]) > min_separation for pose in pose_list]
            if False not in separation_check:
                pose_list.append(sample)
                break

    return pose_list


def randomize_object_pose(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfgs: list[SceneEntityCfg],
    min_separation: float = 0.0,
    pose_range: dict[str, tuple[float, float]] = {},
    max_sample_tries: int = 5000,
):
    if env_ids is None:
        return

    # Randomize poses in each environment independently
    for cur_env in env_ids.tolist():
        pose_list = sample_object_poses(
            num_objects=len(asset_cfgs),
            min_separation=min_separation,
            pose_range=pose_range,
            max_sample_tries=max_sample_tries,
        )

        # Randomize pose for each object
        for i in range(len(asset_cfgs)):
            asset_cfg = asset_cfgs[i]
            asset = env.scene[asset_cfg.name]

            # Write pose to simulation
            pose_tensor = torch.tensor([pose_list[i]], device=env.device)
            positions = pose_tensor[:, 0:3] + env.scene.env_origins[cur_env, 0:3]
            orientations = math_utils.quat_from_euler_xyz(pose_tensor[:, 3], pose_tensor[:, 4], pose_tensor[:, 5])
            asset.write_root_pose_to_sim(
                torch.cat([positions, orientations], dim=-1), env_ids=torch.tensor([cur_env], device=env.device)
            )
            asset.write_root_velocity_to_sim(
                torch.zeros(1, 6, device=env.device), env_ids=torch.tensor([cur_env], device=env.device)
            )


def randomize_rigid_objects_in_focus(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfgs: list[SceneEntityCfg],
    out_focus_state: torch.Tensor,
    min_separation: float = 0.0,
    pose_range: dict[str, tuple[float, float]] = {},
    max_sample_tries: int = 5000,
):
    if env_ids is None:
        return

    # List of rigid objects in focus for each env (dim = [num_envs, num_rigid_objects])
    env.rigid_objects_in_focus = []

    for cur_env in env_ids.tolist():
        # Sample in focus object poses
        pose_list = sample_object_poses(
            num_objects=len(asset_cfgs),
            min_separation=min_separation,
            pose_range=pose_range,
            max_sample_tries=max_sample_tries,
        )

        selected_ids = []
        for asset_idx in range(len(asset_cfgs)):
            asset_cfg = asset_cfgs[asset_idx]
            asset = env.scene[asset_cfg.name]

            # Randomly select an object to bring into focus
            object_id = random.randint(0, asset.num_objects - 1)
            selected_ids.append(object_id)

            # Create object state tensor
            object_states = torch.stack([out_focus_state] * asset.num_objects).to(device=env.device)
            pose_tensor = torch.tensor([pose_list[asset_idx]], device=env.device)
            positions = pose_tensor[:, 0:3] + env.scene.env_origins[cur_env, 0:3]
            orientations = math_utils.quat_from_euler_xyz(pose_tensor[:, 3], pose_tensor[:, 4], pose_tensor[:, 5])
            object_states[object_id, 0:3] = positions
            object_states[object_id, 3:7] = orientations

            asset.write_object_state_to_sim(
                object_state=object_states, env_ids=torch.tensor([cur_env], device=env.device)
            )

        env.rigid_objects_in_focus.append(selected_ids)

"""Simulate communicationo delays and noise in control commands to mimic real-world robotics systems."""
def randomize_control_latency(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    latency_steps_range: tuple[int, int] = (0, 3),  # 0-3 timesteps delay (0-150ms at 20Hz)
):
    """Simulate control latency by delaying action application."""
    # Store this in environment to implement action buffering
    if not hasattr(env, 'action_delay_buffer'):
        env.action_delay_buffer = {}
        env.action_delays = {}
    
    for env_id in env_ids:
        # Sample random delay for this environment
        delay = np.random.randint(latency_steps_range[0], latency_steps_range[1] + 1)
        env.action_delays[env_id.item()] = delay
        env.action_delay_buffer[env_id.item()] = []


def randomize_control_noise(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    action_noise_std: float = 0.01,  # 1% action noise
    velocity_noise_std: float = 0.005,  # Joint velocity noise
):
    """Add noise to control commands to simulate real actuator behavior."""
    robot: Articulation = env.scene[asset_cfg.name]
    
    # Add noise to joint position targets
    if hasattr(robot, '_joint_pos_target'):
        noise = torch.randn_like(robot._joint_pos_target) * action_noise_std
        robot._joint_pos_target += noise
    
    # Add velocity disturbances
    vel_noise = torch.randn(len(env_ids), robot.num_joints, device=env.device) * velocity_noise_std
    current_vel = robot.data.joint_vel[env_ids]
    robot.set_joint_velocity_target(current_vel + vel_noise, env_ids=env_ids)

def randomize_control_frequency(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    frequency_range: tuple[int, int] = (15, 25),  # 15-25 Hz variation around 20Hz
):
    """Simulate variable control frequencies due to computational load."""
    # Store per-environment decimation factors
    if not hasattr(env, 'variable_decimation'):
        env.variable_decimation = {}
    
    for env_id in env_ids:
        target_freq = np.random.randint(*frequency_range)
        # Convert frequency to decimation (sim runs at 100Hz)
        decimation = int(100 / target_freq)
        env.variable_decimation[env_id.item()] = decimation


def set_fixed_object_poses(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfgs: list[SceneEntityCfg],
    fixed_poses: list[dict],
):
    """Set fixed poses for objects.
    
    Args:
        env: The environment.
        env_ids: Environment IDs to reset.
        asset_cfgs: List of asset configurations.
        fixed_poses: List of pose dictionaries with keys 'pos' and 'quat' or 'euler'.
                     Example: [{"pos": [0.4, 0.0, 0.02], "euler": [0, 0, 0]}, ...]
    """
    if env_ids is None:
        return

    # Set poses for each object
    for i, asset_cfg in enumerate(asset_cfgs):
        if i < len(fixed_poses):
            asset = env.scene[asset_cfg.name]
            pose_dict = fixed_poses[i]
            
            # Get position
            pos = pose_dict.get("pos", [0.0, 0.0, 0.0])
            
            # Get orientation (prefer quaternion, fallback to euler)
            if "quat" in pose_dict:
                quat = pose_dict["quat"]
                orientations = torch.tensor([quat], device=env.device)
            elif "euler" in pose_dict:
                euler = pose_dict["euler"]
                orientations = math_utils.quat_from_euler_xyz(
                    torch.tensor([euler[0]], device=env.device),
                    torch.tensor([euler[1]], device=env.device), 
                    torch.tensor([euler[2]], device=env.device)
                )
            else:
                # Default to no rotation
                orientations = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=env.device)
            
            # Apply to all specified environments
            for env_id in env_ids:
                positions = torch.tensor([pos], device=env.device) + env.scene.env_origins[env_id, 0:3]
                asset.write_root_pose_to_sim(
                    torch.cat([positions, orientations], dim=-1), 
                    env_ids=torch.tensor([env_id], device=env.device)
                )
                asset.write_root_velocity_to_sim(
                    torch.zeros(1, 6, device=env.device), 
                    env_ids=torch.tensor([env_id], device=env.device)
                )


