# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os
from dataclasses import MISSING

from isaaclab_assets import HUMANOID_28_CFG,G1_CFG,H1_CFG

from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg,RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.utils import configclass
import isaaclab.sim as sim_utils


@configclass
class HOIEnvCfg(DirectRLEnvCfg):
    # --- 基本 env 配置 ---
    episode_length_s: float = 10.0
    decimation: int = 2

    # 先给占位，具体维度在 env __init__ 里动态重算并刷新
    observation_space = 8
    action_space = 4
    state_space = 0

    early_termination: bool = True
    termination_height: float = 0.5

    reset_strategy: str = "default"   # "default" | "random"（random = 关节位姿随机化）

    # --- 物理/仿真 ---
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        render_interval=decimation,
        physx=PhysxCfg(
            gpu_found_lost_pairs_capacity=2**23,
            gpu_total_aggregate_pairs_capacity=2**23,
        ),
    )

    # --- 场景 ---
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=8.0, replicate_physics=True)

    # --- 椅子 ---
    obj: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Object",
        spawn=sim_utils.UsdFileCfg(
            usd_path="/home/luohy/MyRepository/MyDataSets/Data/Object/chair/3FO3U59GXQ0M/Instance.usda",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                kinematic_enabled=True,   # 如需固定椅子可改 True
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
    )

    # --- 任务参数（坐下->起身） ---
    sit_duration_s: float = 3.0
    stand_duration_s: float = 3.0
    seat_height: float = 0.45
    chair_xy_spawn_range: float = 3.0  # 椅子 XY 随机
    chair_yaw_spawn_range: float = 0.5 # 椅子 Yaw 随机 (rad)

    # 奖励权重（原有）
    w_reach_seat: float = 1.0
    w_sit_height: float = 1.0
    w_seat_contact: float = 0.5
    w_stand_height: float = 1.2
    w_leave_seat: float = 0.6
    w_stability: float = 0.3
    w_action: float = 0.0



@configclass
class HOISmplEnvCfg(HOIEnvCfg):
    # motion_file = os.path.join(MOTIONS_DIR, "humanoid_dance.npz")
    reference_body: str = "torso"
    key_body_names = ["right_hand", "left_hand", "right_foot", "left_foot"]

    # robot
    robot: ArticulationCfg = HUMANOID_28_CFG.replace(prim_path="/World/envs/env_.*/Robot").replace(
        actuators={
            "body": ImplicitActuatorCfg(
                joint_names_expr=[".*"],
                velocity_limit=100.0,
                stiffness=None,   
                damping=None,     
                # stiffness=80.0,   # 先从 60~120 试
                # damping=4.0,      # 先从 2~8 试
            ),
        },
        init_state=ArticulationCfg.InitialStateCfg(
            # pos=(0.0, 0.0, 0.8),
            pos=(0.0, 0.0, 1.3),
            joint_pos={".*": 0.0},
        ),
    )




@configclass
class HOIG1EnvCfg(HOIEnvCfg):
    # motion_file = os.path.join(MOTIONS_DIR, "humanoid_dance.npz")
    reference_body: str = "torso_link"
    key_body_names = ["right_palm_link", "left_palm_link","right_ankle_roll_link", "left_ankle_roll_link"]
    # robot
    robot: ArticulationCfg = G1_CFG.replace(prim_path="/World/envs/env_.*/Robot")

@configclass
class HOIH1EnvCfg(HOIEnvCfg):
    # motion_file = os.path.join(MOTIONS_DIR, "humanoid_dance.npz")
    reference_body: str = "torso_link"
    key_body_names = ["right_elbow_link", "left_elbow_link", "right_ankle_link", "left_ankle_link"]
    # robot
    robot: ArticulationCfg = H1_CFG.replace(prim_path="/World/envs/env_.*/Robot")