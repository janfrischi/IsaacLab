# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, DeformableObjectCfg, RigidObjectCfg
from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.noise.noise_cfg import GaussianNoiseCfg

from . import mdp

##
# Scene definition
##

# All the MISSING variables are defined in the joint_pos_env_cfg.py file "Specific configurations for the robot"
@configclass
class ObjectTableSceneCfg(InteractiveSceneCfg):
    """Configuration for the lift scene with a robot and an object."""
    # Define the robot and object used in the scene
    robot: ArticulationCfg = MISSING
    object: RigidObjectCfg | DeformableObjectCfg = MISSING
    # Define all the frames used in the scene
    ee_frame: FrameTransformerCfg = MISSING
    object_frame: FrameTransformerCfg = MISSING
    robot_frame: FrameTransformerCfg = MISSING
    
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.5, 0, 0], rot=[0.707, 0, 0, 0.707]),
        spawn=UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd"),
    )

    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0, 0, -1.05]),
        spawn=GroundPlaneCfg(),
    )

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )

##
# MDP settings
##

# Define target pose for the object
@configclass
class CommandsCfg:
    """Command terms for the MDP."""

    object_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name=MISSING, # Define in joint_pos_env_cfg.py
        resampling_time_range=(5.0, 5.0),
        debug_vis=True,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(0.5, 0.5), pos_y=(0.0, 0.0), pos_z=(0.5, 0.5),
            roll=(3.14159, 3.14159), pitch=(0.0, 0.0), yaw=(0.0, 0.0)
        ),
    )

@configclass
class ActionsCfg:
    """Action specifications for the MDP."""
    # Allow for both joint position and Differential inverse kinematics actions
    arm_action: mdp.JointPositionActionCfg | mdp.DifferentialInverseKinematicsActionCfg = MISSING
    gripper_action: mdp.BinaryJointPositionActionCfg = MISSING

@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)

        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            #noise=GaussianNoiseCfg(mean=0.0, std=0.01)
        
        )

        object_position = ObsTerm(
            func=mdp.object_position_in_robot_frame,
            #noise=GaussianNoiseCfg(mean=0.0, std=0.01)
        )
        
        object_orientation = ObsTerm(
            func=mdp.object_orientation_in_robot_frame,
            #noise=GaussianNoiseCfg(mean=0.0, std=0.02)
        )

        target_object_position = ObsTerm(func=mdp.generated_commands, params={"command_name": "object_pose"})

        actions = ObsTerm(func=mdp.last_action)
        
        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()

# Reset the object position and orientation
@configclass
class EventCfg:
    """Configuration for events."""

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")
    reset_object_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-0.3, 0.3),
                "y": (-0.35, 0.35),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("object", body_names="Object"),
        },
    )
    
    
@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    reach = RewTerm(
        func=mdp.object_ee_distance,
        params={
            "std":               0.1,
            "bonus_threshold":   0.02,
            "bonus_weight":      0.3,
            "object_cfg":        SceneEntityCfg("object"),
            "ee_frame_cfg":      SceneEntityCfg("ee_frame"),
        },
        weight=7.0,
    )

    lifted = RewTerm(
        func=mdp.object_is_lifted,
        params={
            "minimal_height": 0.04,
            "lift_width":     0.01,
            "object_cfg":     SceneEntityCfg("object"),
        },
        weight=20.0,
    )

    # Dense reward for lifting
    lift_shaping = RewTerm(
        func=mdp.potential_based_lift,
        params={"minimal_height": 0.04, "object_cfg": SceneEntityCfg("object")},
        weight=9.1,
    )
    
    pos_reward = RewTerm(
        func=mdp.object_goal_distance_pos,
        params={
            "std":           0.05, #changed from 0.3
            "minimal_height":0.04,
            "lift_width":    0.01,
            "command_name":  "object_pose",
            "robot_cfg":      SceneEntityCfg("robot"),
            "object_cfg":     SceneEntityCfg("object"),
        },
        weight=15.0, #changed from 10.0
    )
    
    ori_reward = RewTerm(
        func=mdp.ee_goal_distance_ori,
        params={
            "std":          0.2,
            "command_name":"object_pose",
            "robot_cfg":      SceneEntityCfg("robot"),
            "ee_frame_cfg": SceneEntityCfg("ee_frame"),
        },
        weight=0.0,   # Adjust as needed based on other rewards
    )
    
    # Action rate penalty -> Penalize only arm joint actions
    action_rate = RewTerm(func=mdp.action_rate_l2_no_gripper, weight=-1e-4)

    # joint velocity penalty
    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-1e-4,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    
    # joint acceleration penalty
    joint_acc = RewTerm(
        func=mdp.joint_acc_l2,
        weight=-1e-9, # Example: start with a small weight
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    

@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    object_dropping = DoneTerm(
        func=mdp.root_height_below_minimum, params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("object")}
    )
    

@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""
    
    # Start with a high weight for the reach term and gradually reduce it
    reach = CurrTerm(
        # num_steps = iterations * num_steps_per_env
        func=mdp.gradually_modify_reward_weight, params={
            "term_name": "reach",
            "start_weight": 7.0,
            "end_weight": 0.0, # TODO: Check whether positive reward is needed
            "num_steps": 2000 * 24,
            "curve_type": "linear",
            "start_step": 100 * 24,
        }
    )
    
    # Ramp and decay the positive reward for the object position
    pos_reward_ramp = CurrTerm(
        func=mdp.gradually_modify_reward_weight, params={
            "term_name": "pos_reward",
            "start_weight": 10.0,
            "end_weight": 30.0,
            "num_steps": 2000 * 24,
            "curve_type": "linear",
            "start_step": 1000 * 24,
        }
    )
    
    pos_reward_decay = CurrTerm(
        func=mdp.gradually_modify_reward_weight, params={
            "term_name": "pos_reward",
            "start_weight": 30.0,
            "end_weight": 5.0,
            "num_steps": 2000 * 24,
            "curve_type": "linear",
            "start_step": 2500 * 24,
        }
    )
    
    ori_reward = CurrTerm(
        func=mdp.gradually_modify_reward_weight, params={
            "term_name": "ori_reward",
            "start_weight": 0.0,
            "end_weight": 30.0, # We raised this from 20.0
            "num_steps": 3000 * 24,
            "curve_type": "linear",
            "start_step": 250 * 24, # before 500
        }
    )
    
    lift_shaping = CurrTerm(
        func= mdp.gradually_modify_reward_weight,
        params={
            "term_name": "lift_shaping",
            "start_weight": 9.1, # High initial weight
            "end_weight": 0.0, # Gradually reduce to 0
            "num_steps": 1500 * 24,
            "curve_type": "linear",
            "start_step": 1500 * 24,
        }
    )
    
    # Penalize the action rate
    action_rate = CurrTerm(
        func=mdp.gradually_modify_reward_weight, params={
            "term_name": "action_rate",
            "start_weight": -1e-4,
            "end_weight": -10.0,
            "num_steps": 2000 * 24,
            "curve_type": "exp",
            "start_step": 2000 * 24,
        }
    )
    
    # Penalize the joint velocity
    joint_vel = CurrTerm(
        func=mdp.gradually_modify_reward_weight, params={
            "term_name": "joint_vel",
            "start_weight": -1e-4,
            "end_weight": -1.0,
            "num_steps": 2000 * 24,
            "curve_type": "exp",
            "start_step": 2000 * 24,
        }
    )
    
##
# Environment configuration
##


@configclass
class LiftEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the lifting environment."""

    scene: ObjectTableSceneCfg = ObjectTableSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        self.decimation = 2
        self.episode_length_s = 5.0
        self.sim.dt = 0.01
        self.sim.render_interval = self.decimation
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625