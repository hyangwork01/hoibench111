# Copyright (c) 2022-2025
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

from isaaclab.actuators import ImplicitActuatorCfg
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.utils import configclass
from .base import HOIEnvCfg
from isaaclab_assets import HUMANOID_28_CFG, G1_CFG, H1_CFG


@configclass
class ClawEnvCfg(HOIEnvCfg):

                                          
    table: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Table",
        spawn=sim_utils.UsdFileCfg(
            usd_path="/home/luohy/MyRepository/MyDataSets/Data/Object/table/3FO3WQT7FQR5/Instance.usda",  
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                kinematic_enabled=True,                                 
                angular_damping=0.0,
                linear_damping=0.0,
                max_linear_velocity=100.0,
                max_angular_velocity=100.0,
                retain_accelerations=False,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
                contact_offset=0.002,
                rest_offset=0.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(
                                             
                density=2000.0
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
                                              
            pos=(1.5, 1.5, 0.75),
            rot=(0.0, 0.0, 0.0, 1.0),  # wxyz
        ),
    )

                                           
    toy: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Toy",
        spawn=sim_utils.UsdFileCfg(
            usd_path="/home/luohy/MyRepository/MyDataSets/Data/Object/Toy/3FO3NNCTMYLD/Instance.usda", 
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                kinematic_enabled=False,                       
                angular_damping=0.05,
                linear_damping=0.01,
                max_linear_velocity=100.0,
                max_angular_velocity=100.0,
                retain_accelerations=False,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
                contact_offset=0.002,
                rest_offset=0.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(
                density=500.0               
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(1.5, 1.5, 1.0),                             
            rot=(0.0, 0.0, 0.0, 1.0),      # wxyz
        ),
    )

                                      
    goal: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Goal",
        spawn=sim_utils.SphereCfg(
            radius=0.03,
                                                       
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True, kinematic_enabled=True,
                max_linear_velocity=0.0, max_angular_velocity=0.0,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(1.0, 0.0, 0.0), roughness=0.3, metallic=0.0
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0, 0.0, 1.0),
            rot=(1.0, 0.0, 0.0, 0.0),  # wxyz
        ),
    )
                                 


@configclass
class ClawSmplEnvCfg(ClawEnvCfg):
    reference_body: str = "torso"
    contact_body: str = "right_hand"
    key_body_names = ["right_hand", "left_hand", "right_foot", "left_foot"]
    robot: ArticulationCfg = HUMANOID_28_CFG.replace(
        prim_path="/World/envs/env_.*/Robot"
    ).replace(
        actuators={
            "body": ImplicitActuatorCfg(
                joint_names_expr=[".*"],
                velocity_limit=100.0,
                stiffness=None,
                damping=None,
            ),
        },
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 1.3),
            joint_pos={".*": 0.0},
        ),
    )


@configclass
class ClawG1EnvCfg(ClawEnvCfg):
    reference_body: str = "torso_link"
    contact_body: str = "right_palm_link"
    key_body_names = ["right_palm_link", "left_palm_link", "right_ankle_roll_link", "left_ankle_roll_link"]
    robot: ArticulationCfg = G1_CFG.replace(prim_path="/World/envs/env_.*/Robot")


@configclass
class ClawH1EnvCfg(ClawEnvCfg):
    reference_body: str = "torso_link"
    contact_body: str = "right_elbow_link"
    key_body_names = ["right_elbow_link", "left_elbow_link", "right_ankle_link", "left_ankle_link"]
    robot: ArticulationCfg = H1_CFG.replace(prim_path="/World/envs/env_.*/Robot")
