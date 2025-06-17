# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Franka Emika robots.

The following configurations are available:

* :obj:`FRANKA_PANDA_CFG`: Franka Emika Panda robot with Panda hand
* :obj:`FRANKA_PANDA_HIGH_PD_CFG`: Franka Emika Panda robot with Panda hand with stiffer PD control
* :obj:`FRANKA_PANDA_REAL_ROBOT_CFG`: Franka Emika Panda robot with Panda hand with real robot joint stiffnesses

Reference: https://github.com/frankaemika/franka_ros
"""

import isaaclab.sim as sim_utils
import numpy as np
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

##
# Configuration
##

FRANKA_PANDA_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/FrankaEmika/panda_instanceable.usd",
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=0, fix_root_link=True
        ),
        # collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "panda_joint1": 0.0,
            "panda_joint2": -0.569,
            "panda_joint3": 0.0,
            "panda_joint4": -2.810,
            "panda_joint5": 0.0,
            "panda_joint6": 3.037,
            "panda_joint7": 0.741,
            "panda_finger_joint.*": 0.04,
        },
    ),
    actuators={
        "panda_shoulder": ImplicitActuatorCfg(
            joint_names_expr=["panda_joint[1-4]"],
            effort_limit=87.0,
            velocity_limit=2.175,
            stiffness=80.0,
            damping=4.0,
        ),
        "panda_forearm": ImplicitActuatorCfg(
            joint_names_expr=["panda_joint[5-7]"],
            effort_limit=12.0,
            velocity_limit=2.61,
            stiffness=80.0,
            damping=4.0,
        ),
        "panda_hand": ImplicitActuatorCfg(
            joint_names_expr=["panda_finger_joint.*"],
            effort_limit=200.0,
            velocity_limit=0.2,
            stiffness=2e3,
            damping=1e2,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)
"""Configuration of Franka Emika Panda robot."""


FRANKA_PANDA_HIGH_PD_CFG = FRANKA_PANDA_CFG.copy()
FRANKA_PANDA_HIGH_PD_CFG.spawn.rigid_props.disable_gravity = True
FRANKA_PANDA_HIGH_PD_CFG.actuators["panda_shoulder"].stiffness = 400.0
FRANKA_PANDA_HIGH_PD_CFG.actuators["panda_shoulder"].damping = 80.0
FRANKA_PANDA_HIGH_PD_CFG.actuators["panda_forearm"].stiffness = 400.0
FRANKA_PANDA_HIGH_PD_CFG.actuators["panda_forearm"].damping = 80.0
"""Configuration of Franka Emika Panda robot with stiffer PD control.

This configuration is useful for task-space control using differential IK.
"""

"""Configuration of Franka Emika Panda robot matching real robot joint stiffnesses.
This configuration uses the exact joint-specific stiffness values used on the real robot:
kp = [400, 400, 400, 300, 60, 40, 10] N⋅m/rad
"""
FRANKA_PANDA_REAL_ROBOT_CFG = FRANKA_PANDA_CFG.copy()
FRANKA_PANDA_REAL_ROBOT_CFG.spawn.rigid_props.disable_gravity = True  # Match typical robot control
FRANKA_PANDA_REAL_ROBOT_CFG.actuators = {
    # Joints 1-4: Higher stiffness (shoulder/elbow region)
    "panda_joint1": ImplicitActuatorCfg(
        joint_names_expr=["panda_joint1"],
        effort_limit=87.0,
        velocity_limit=2.175,
        stiffness=400.0,  # Match your kp[0]
        damping=2.5*np.sqrt(400.0),     # Reasonable damping (10% of stiffness)
    ),
    "panda_joint2": ImplicitActuatorCfg(
        joint_names_expr=["panda_joint2"],
        effort_limit=87.0,
        velocity_limit=2.175,
        stiffness=400.0,  # Match your kp[1]
        damping=2.5*np.sqrt(400.0),  
    ),
    "panda_joint3": ImplicitActuatorCfg(
        joint_names_expr=["panda_joint3"],
        effort_limit=87.0,
        velocity_limit=2.175,
        stiffness=400.0,  # Match your kp[2]
        damping=2.5*np.sqrt(400.0),
    ),
    "panda_joint4": ImplicitActuatorCfg(
        joint_names_expr=["panda_joint4"],
        effort_limit=87.0,
        velocity_limit=2.175,
        stiffness=300.0,  # Match your kp[3]
        damping=2.5*np.sqrt(300.0),
    ),
    # Joints 5-7: Lower stiffness (wrist region)
    "panda_joint5": ImplicitActuatorCfg(
        joint_names_expr=["panda_joint5"],
        effort_limit=12.0,
        velocity_limit=2.61,
        stiffness=60.0,   # Match your kp[4]
        damping=2.5*np.sqrt(60.0),
    ),
    "panda_joint6": ImplicitActuatorCfg(
        joint_names_expr=["panda_joint6"],
        effort_limit=12.0,
        velocity_limit=2.61,
        stiffness=40.0,   # Match your kp[5]
        damping=2.5*np.sqrt(40.0),
    ),
    "panda_joint7": ImplicitActuatorCfg(
        joint_names_expr=["panda_joint7"],
        effort_limit=12.0,
        velocity_limit=2.61,
        stiffness=10.0,   # Match your kp[6]
        damping=2.5*np.sqrt(10.0),
    ),
    # Gripper
    "panda_hand": ImplicitActuatorCfg(
        joint_names_expr=["panda_finger_joint.*"],
        effort_limit=200.0,
        velocity_limit=0.2,
        stiffness=2e3,
        damping=1e2,
    ),
}

