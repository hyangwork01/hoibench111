# Copyright (c) 2022-2025
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

from isaaclab_assets import HUMANOID_28_CFG, G1_CFG, H1_CFG
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
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

    # 观测/动作占位（实际在 env 里重算）
    observation_space = 8
    action_space = 4
    state_space = 0

    # 历史帧堆叠
    history_len: int = 5

    # 早停（防“躺平刷分”）
    early_termination: bool = True
    termination_height: float = 0.3        # 抬高高度阈值

    reset_strategy: str = "default"   # "default" | "random"

    # --- 物理/仿真 ---
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 60,
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
    )

    # --- 任务参数（坐下->起身） ---
    sit_duration_s: float = 3.0
    stand_duration_s: float = 3.0
    seat_height: float = 0.45

    # 课程：先易后难（学会靠近/站立后再调大）
    chair_xy_spawn_range: float = 1.0       # 先 1.0，学会后逐步涨到 3.0
    chair_yaw_spawn_range: float = 0.2      # 同上，之后再到 0.5

    # 阶段门控与距离
    approach_dist_th: float = 0.6
    leave_dist_th: float = 0.6
    blend_k: float = 6.0
    away_beta: float = 1.5

    # 指定区域（靠近区）与进入奖励
    approach_zone_radius: float = 0.35
    enter_zone_bonus: float = 1.5

    # 站立姿态奖励
    stand_height_offset: float = 0.35
    stand_posture_sigma: float = 0.08
    w_stand_posture: float = 0.4

    # —— Locomotion 风格：探索与代价 —— #
    progress_weight: float = 2.0            # Δdistance 进度奖励
    alive_reward_scale: float = 1.5         # 存活奖励基线（配合upright门控）
    alive_upright_min: float = 0.6          # up_proj 低于此值，alive 打折/为零
    tilt_termination_cos: float = 0.3       # 倾斜过大直接终止（up_proj < 0.3）
    stuck_vel_threshold: float = 0.05       # “朝目标速度”过小视为无进展
    stuck_time_s: float = 1.0               # 无进展累计时长阈值（判卡住）

    # 早期探索噪声（按回合前几秒注入到目标关节位姿）
    exploration_noise_std: float = 0.10
    exploration_noise_until_s: float = 8.0

    # 能耗/动作正则（先小，学会走再加严）
    energy_cost_scale: float = 0.01
    actions_cost_scale: float = 0.005
    w_action_rate: float = 0.01

    # 奖励权重（原有）
    w_reach_seat: float = 1.0
    w_sit_height: float = 1.0
    w_seat_contact: float = 0.5
    w_stand_height: float = 1.2
    w_leave_seat: float = 0.6
    w_stability: float = 0.3

    # 四块缩放
    w_block_approach: float = 1.0
    w_block_sit: float = 1.0
    w_block_stand: float = 1.0
    w_block_leave: float = 1.0

    # 行走控制参数
    goal_threshold: float = 0.25
    goal_bonus: float = 3.0
    heading_weight_walk: float = 0.5
    up_weight_walk: float = 0.3
    target_speed: float = 1.0
    speed_sigma: float = 0.4
    speed_weight: float = 0.5
    lateral_vel_weight: float = 0.05
    vertical_vel_weight: float = 0.05
    angvel_weight: float = 0.05

    # 观测归一化
    dof_vel_scale: float = 0.1


@configclass
class HOISmplEnvCfg(HOIEnvCfg):
    reference_body: str = "torso"
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
class HOIG1EnvCfg(HOIEnvCfg):
    reference_body: str = "torso_link"
    key_body_names = ["right_palm_link", "left_palm_link", "right_ankle_roll_link", "left_ankle_roll_link"]
    robot: ArticulationCfg = G1_CFG.replace(prim_path="/World/envs/env_.*/Robot")


@configclass
class HOIH1EnvCfg(HOIEnvCfg):
    reference_body: str = "torso_link"
    key_body_names = ["right_elbow_link", "left_elbow_link", "right_ankle_link", "left_ankle_link"]
    robot: ArticulationCfg = H1_CFG.replace(prim_path="/World/envs/env_.*/Robot")
