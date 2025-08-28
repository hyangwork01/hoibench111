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
class TouchEnvCfg(HOIEnvCfg):

                                      
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
class TouchSmplEnvCfg(TouchEnvCfg):
    reference_body: str = "torso"
    contact_body: str = "right_hand"
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
class TouchG1EnvCfg(TouchEnvCfg):
    reference_body: str = "torso_link"
    contact_body: str = "right_palm_link"
    robot: ArticulationCfg = G1_CFG.replace(prim_path="/World/envs/env_.*/Robot")


@configclass
class TouchH1EnvCfg(TouchEnvCfg):
    reference_body: str = "torso_link"
    contact_body: str = "right_elbow_link"
    robot: ArticulationCfg = H1_CFG.replace(prim_path="/World/envs/env_.*/Robot")

