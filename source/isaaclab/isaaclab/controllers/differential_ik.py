# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.utils.math import apply_delta_pose, compute_pose_error

if TYPE_CHECKING:
    from .differential_ik_cfg import DifferentialIKControllerCfg


class DifferentialIKController:
    """Differential inverse kinematics (IK) controller with optional moving average filtering.

    This controller is based on the concept of differential inverse kinematics [1, 2] which is a method for computing
    the change in joint positions that yields the desired change in pose.

    To deal with singularity in Jacobian, the following methods are supported for computing inverse of the Jacobian:

    - "pinv": Moore-Penrose pseudo-inverse
    - "svd": Adaptive singular-value decomposition (SVD)
    - "trans": Transpose of matrix
    - "dls": Damped version of Moore-Penrose pseudo-inverse (also called Levenberg-Marquardt)


    .. caution::
        The controller does not assume anything about the frames of the current and desired end-effector pose,
        or the joint-space velocities. It is up to the user to ensure that these quantities are given
        in the correct format.

    Reference:

    1. `Robot Dynamics Lecture Notes <https://ethz.ch/content/dam/ethz/special-interest/mavt/robotics-n-intelligent-systems/rsl-dam/documents/RobotDynamics2017/RD_HS2017script.pdf>`_
       by Marco Hutter (ETH Zurich)
    2. `Introduction to Inverse Kinematics <https://www.cs.cmu.edu/~15464-s13/lectures/lecture6/iksurvey.pdf>`_
       by Samuel R. Buss (University of California, San Diego)

    """

    def __init__(self, cfg: DifferentialIKControllerCfg, num_envs: int, device: str):
        """Initialize the controller.

        Args:
            cfg: The configuration for the controller.
            num_envs: The number of environments.
            device: The device to use for computations.
        """
        # store inputs
        self.cfg = cfg
        self.num_envs = num_envs
        self._device = device
        
        # create buffers
        self.ee_pos_des = torch.zeros(self.num_envs, 3, device=self._device)
        self.ee_quat_des = torch.zeros(self.num_envs, 4, device=self._device)
        
        # NEW: Moving average filter state buffers - only create if needed
        if self.cfg.use_moving_average_filter:
            self._filtered_pos_target = torch.zeros(self.num_envs, 3, device=self._device)
            self._filtered_quat_target = torch.zeros(self.num_envs, 4, device=self._device)
            # Initialize with identity quaternion
            self._filtered_quat_target[:, 0] = 1.0  # w component
            self._filter_initialized = torch.zeros(self.num_envs, dtype=torch.bool, device=self._device)
        else:
            # Set to None when not using filter to avoid attribute errors
            self._filtered_pos_target = None
            self._filtered_quat_target = None
            self._filter_initialized = None
        
        # -- input command
        self._command = torch.zeros(self.num_envs, self.action_dim, device=self._device)

    @property
    def action_dim(self) -> int:
        """Dimension of the controller's input command."""
        if self.cfg.command_type == "position":
            return 3  # (x, y, z)
        elif self.cfg.command_type == "pose" and self.cfg.use_relative_mode:
            return 6  # (dx, dy, dz, droll, dpitch, dyaw)
        else:
            return 7  # (x, y, z, qw, qx, qy, qz)

    """
    Operations.
    """
    def reset(self, env_ids: torch.Tensor = None):
        """Reset the internals."""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self._device)
        
        # Reset filter state for specified environments - only if filter is enabled
        if self.cfg.use_moving_average_filter and self._filter_initialized is not None:
            self._filter_initialized[env_ids] = False

    def set_command(
        self, command: torch.Tensor, ee_pos: torch.Tensor | None = None, ee_quat: torch.Tensor | None = None
    ):
        """Set target end-effector pose command with optional filtering.

        Based on the configured command type and relative mode, the method computes the desired end-effector pose.
        It is up to the user to ensure that the command is given in the correct frame. The method only
        applies the relative mode if the command type is ``position_rel`` or ``pose_rel``.

        Args:
            command: The input command in shape (N, 3) or (N, 6) or (N, 7).
            ee_pos: The current end-effector position in shape (N, 3).
                This is only needed if the command type is ``position_rel`` or ``pose_rel``.
            ee_quat: The current end-effector orientation (w, x, y, z) in shape (N, 4).
                This is only needed if the command type is ``position_*`` or ``pose_rel``.

        Raises:
            ValueError: If the command type is ``position_*`` and :attr:`ee_quat` is None.
            ValueError: If the command type is ``position_rel`` and :attr:`ee_pos` is None.
            ValueError: If the command type is ``pose_rel`` and either :attr:`ee_pos` or :attr:`ee_quat` is None.
        """
        # store command
        self._command[:] = command
        
        # compute the raw desired end-effector pose
        if self.cfg.command_type == "position":
            if ee_quat is None:
                raise ValueError("End-effector orientation can not be None for `position_*` command type!")
            if self.cfg.use_relative_mode:
                if ee_pos is None:
                    raise ValueError("End-effector position can not be None for `position_rel` command type!")
                raw_pos_des = ee_pos + self._command
                raw_quat_des = ee_quat
            else:
                raw_pos_des = self._command
                raw_quat_des = ee_quat
        else:
            if self.cfg.use_relative_mode:
                if ee_pos is None or ee_quat is None:
                    raise ValueError("Neither end-effector position nor orientation can be None for `pose_rel` command type!")
                raw_pos_des, raw_quat_des = apply_delta_pose(ee_pos, ee_quat, self._command)
            else:
                raw_pos_des = self._command[:, 0:3]
                raw_quat_des = self._command[:, 3:7]

        # Apply moving average filter if enabled and properly initialized
        if self.cfg.use_moving_average_filter and self._filtered_pos_target is not None:
            self.ee_pos_des, self.ee_quat_des = self._apply_moving_average_filter(raw_pos_des, raw_quat_des)
        else:
            self.ee_pos_des[:] = raw_pos_des
            self.ee_quat_des[:] = raw_quat_des

    def _batch_quat_slerp(self, q1: torch.Tensor, q2: torch.Tensor, t: float) -> torch.Tensor:
        """Batch quaternion spherical linear interpolation (SLERP).
        
        Args:
            q1: Source quaternions [N, 4] (w, x, y, z)
            q2: Target quaternions [N, 4] (w, x, y, z) 
            t: Interpolation factor [0, 1]
            
        Returns:
            Interpolated quaternions [N, 4]
        """
        # Ensure quaternions are normalized
        q1 = q1 / torch.norm(q1, dim=-1, keepdim=True)
        q2 = q2 / torch.norm(q2, dim=-1, keepdim=True)
        
        # Compute dot product for each quaternion pair
        dot = torch.sum(q1 * q2, dim=-1, keepdim=True)  # [N, 1]
        
        # If dot product is negative, negate one quaternion to take shorter path
        q2 = torch.where(dot < 0.0, -q2, q2)
        dot = torch.where(dot < 0.0, -dot, dot)
        
        # If quaternions are very close, use linear interpolation
        threshold = 0.9995
        close_mask = (dot > threshold)
        
        # Linear interpolation for close quaternions
        result_linear = (1 - t) * q1 + t * q2
        result_linear = result_linear / torch.norm(result_linear, dim=-1, keepdim=True)
        
        # SLERP for distant quaternions
        theta = torch.acos(torch.clamp(dot, -1.0, 1.0))  # [N, 1]
        sin_theta = torch.sin(theta)
        
        # Avoid division by zero
        safe_sin_theta = torch.where(sin_theta.abs() < 1e-6, torch.ones_like(sin_theta), sin_theta)
        
        w1 = torch.sin((1 - t) * theta) / safe_sin_theta  # [N, 1]
        w2 = torch.sin(t * theta) / safe_sin_theta        # [N, 1]
        
        result_slerp = w1 * q1 + w2 * q2
        
        # Choose between linear and slerp based on closeness
        result = torch.where(close_mask, result_linear, result_slerp)
        
        return result / torch.norm(result, dim=-1, keepdim=True)

    def _apply_moving_average_filter(self, target_pos: torch.Tensor, target_quat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply moving average filter to target pose."""
        # Safety check - ensure filter buffers exist
        if self._filtered_pos_target is None or self._filtered_quat_target is None or self._filter_initialized is None:
            return target_pos.clone(), target_quat.clone()
        
        # Initialize filter state on first call
        uninitialized_mask = ~self._filter_initialized
        if torch.any(uninitialized_mask):
            self._filtered_pos_target[uninitialized_mask] = target_pos[uninitialized_mask]
            self._filtered_quat_target[uninitialized_mask] = target_quat[uninitialized_mask]
            self._filter_initialized[uninitialized_mask] = True

        # Apply position filtering: position_d_ = (1.0 - α) * position_d_ + α * target
        alpha_pos = self.cfg.position_filter_factor
        self._filtered_pos_target = (1.0 - alpha_pos) * self._filtered_pos_target + alpha_pos * target_pos

        # Apply orientation filtering using batch SLERP: orientation_d_.slerp(α, target)
        alpha_quat = self.cfg.orientation_filter_factor
        self._filtered_quat_target = self._batch_quat_slerp(self._filtered_quat_target, target_quat, alpha_quat)

        return self._filtered_pos_target.clone(), self._filtered_quat_target.clone()

    def compute(
        self, ee_pos: torch.Tensor, ee_quat: torch.Tensor, jacobian: torch.Tensor, joint_pos: torch.Tensor
    ) -> torch.Tensor:
        """Computes the target joint positions that will yield the desired end effector pose.

        Args:
            ee_pos: The current end-effector position in shape (N, 3).
            ee_quat: The current end-effector orientation in shape (N, 4).
            jacobian: The geometric jacobian matrix in shape (N, 6, num_joints).
            joint_pos: The current joint positions in shape (N, num_joints).

        Returns:
            The target joint positions commands in shape (N, num_joints).
        """
        # 1. Compute the delta in joint-space
        if "position" in self.cfg.command_type:
            position_error = self.ee_pos_des - ee_pos
            jacobian_pos = jacobian[:, 0:3]
            # 2. Map pose error to joint space using Jacobian
            delta_joint_pos = self._compute_delta_joint_pos(delta_pose=position_error, jacobian=jacobian_pos)
        else:
            position_error, axis_angle_error = compute_pose_error(
                ee_pos, ee_quat, self.ee_pos_des, self.ee_quat_des, rot_error_type="axis_angle"
            )
            pose_error = torch.cat((position_error, axis_angle_error), dim=1)
            delta_joint_pos = self._compute_delta_joint_pos(delta_pose=pose_error, jacobian=jacobian)
        # 3. Return target joint positions
        return joint_pos + delta_joint_pos

    """
    Helper functions.
    """
    def _compute_delta_joint_pos(self, delta_pose: torch.Tensor, jacobian: torch.Tensor) -> torch.Tensor:
        """Computes the change in joint position that yields the desired change in pose.

        The method uses the Jacobian mapping from joint-space velocities to end-effector velocities
        to compute the delta-change in the joint-space that moves the robot closer to a desired
        end-effector position.

        Args:
            delta_pose: The desired delta pose in shape (N, 3) or (N, 6).
            jacobian: The geometric jacobian matrix in shape (N, 3, num_joints) or (N, 6, num_joints).

        Returns:
            The desired delta in joint space. Shape is (N, num-jointsß).
        """
        if self.cfg.ik_params is None:
            raise RuntimeError(f"Inverse-kinematics parameters for method '{self.cfg.ik_method}' is not defined!")
        
        # compute the delta in joint-space
        if self.cfg.ik_method == "pinv":  # Jacobian pseudo-inverse
            k_val = self.cfg.ik_params["k_val"]
            jacobian_pinv = torch.linalg.pinv(jacobian)
            delta_joint_pos = k_val * jacobian_pinv @ delta_pose.unsqueeze(-1)
            delta_joint_pos = delta_joint_pos.squeeze(-1)
        elif self.cfg.ik_method == "svd":  # adaptive SVD
            k_val = self.cfg.ik_params["k_val"]
            min_singular_value = self.cfg.ik_params["min_singular_value"]
            U, S, Vh = torch.linalg.svd(jacobian)
            S_inv = 1.0 / S
            S_inv = torch.where(S > min_singular_value, S_inv, torch.zeros_like(S_inv))
            jacobian_pinv = (
                torch.transpose(Vh, dim0=1, dim1=2)[:, :, :6]
                @ torch.diag_embed(S_inv)
                @ torch.transpose(U, dim0=1, dim1=2)
            )
            delta_joint_pos = k_val * jacobian_pinv @ delta_pose.unsqueeze(-1)
            delta_joint_pos = delta_joint_pos.squeeze(-1)
        elif self.cfg.ik_method == "trans":  # Jacobian transpose
            k_val = self.cfg.ik_params["k_val"]
            jacobian_T = torch.transpose(jacobian, dim0=1, dim1=2)
            delta_joint_pos = k_val * jacobian_T @ delta_pose.unsqueeze(-1)
            delta_joint_pos = delta_joint_pos.squeeze(-1)
        elif self.cfg.ik_method == "dls":  # damped least squares
            lambda_val = self.cfg.ik_params["lambda_val"]
            jacobian_T = torch.transpose(jacobian, dim0=1, dim1=2)
            lambda_matrix = (lambda_val**2) * torch.eye(n=jacobian.shape[1], device=self._device)
            delta_joint_pos = (
                jacobian_T @ torch.inverse(jacobian @ jacobian_T + lambda_matrix) @ delta_pose.unsqueeze(-1)
            )
            delta_joint_pos = delta_joint_pos.squeeze(-1)
        else:
            raise ValueError(f"Unsupported inverse-kinematics method: {self.cfg.ik_method}")

        return delta_joint_pos
