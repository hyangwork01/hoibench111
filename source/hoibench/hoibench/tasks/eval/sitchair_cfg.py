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
class SitchairEnvCfg(HOIEnvCfg):

    obj: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Object",
        spawn=sim_utils.UsdFileCfg(
            usd_path="/home/luohy/MyRepository/MyDataSets/Data/Object/chair/3FO3U59GXQ0M/Instance.usda",
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
            mass_props=sim_utils.MassPropertiesCfg(density=1000.0),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(2.0, 2.0, 0.5),
            rot=(0.0, 0.0, 0.0, 1.0),
        ),

    )

    

@configclass
class SitchairSmplEnvCfg(SitchairEnvCfg):
    reference_body: str = "torso"
    contact_body: str = "pelvis"
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
class SitchairG1EnvCfg(SitchairEnvCfg):
    reference_body: str = "torso_link"
    contact_body: str = "pelvis"
    key_body_names = ["right_palm_link", "left_palm_link", "right_ankle_roll_link", "left_ankle_roll_link"]
    robot: ArticulationCfg = G1_CFG.replace(prim_path="/World/envs/env_.*/Robot")


@configclass
class SitchairH1Cfg(SitchairEnvCfg):
    reference_body: str = "torso_link"
    contact_body: str = "pelvis"
    key_body_names = ["right_elbow_link", "left_elbow_link", "right_ankle_link", "left_ankle_link"]
    robot: ArticulationCfg = H1_CFG.replace(prim_path="/World/envs/env_.*/Robot")

