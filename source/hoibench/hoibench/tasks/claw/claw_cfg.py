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
    """Claw 任务环境配置：包含一张桌子（table）与一个可移动玩具（toy）。"""

    # 桌子：仅作为支撑/放置面；可见、可碰撞、通常设为运动学体（不受力学驱动）
    table: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Table",
        spawn=sim_utils.UsdFileCfg(
            usd_path="/home/luohy/MyRepository/MyDataSets/Data/Object/table/3FO3WQT7FQR5/Instance.usda",  
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                kinematic_enabled=True,     # 作为“静态几何”使用（可通过根姿态移动，但不受力学）
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
                # 桌子作为大件，质量/密度给大一点以避免被小物件“顶飞”
                density=2000.0
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            # 放在每个 env 的示例位置；具体以你的 USD 尺寸为准再调整
            pos=(1.5, 1.5, 0.75),
            rot=(0.0, 0.0, 0.0, 1.0),  # wxyz
        ),
    )

    # 玩具：作为被抓取/推动的目标物；可碰撞、动态刚体（非 kinematic）
    toy: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Toy",
        spawn=sim_utils.UsdFileCfg(
            usd_path="/home/luohy/MyRepository/MyDataSets/Data/Object/Toy/3FO3NNCTMYLD/Instance.usda", 
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                kinematic_enabled=False,    # 让它受物理驱动（可被抓/推/移动）
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
                density=500.0  # 适中；可按真实玩具调整
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(1.5, 1.5, 1.0),           # 例：在桌面上方一些，避免初始相交
            rot=(0.0, 0.0, 0.0, 1.0),      # wxyz
        ),
    )

    # --- 目标标记：红色小球（仅可见、不可碰撞、运动学体） ---
    goal: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Goal",
        spawn=sim_utils.SphereCfg(
            radius=0.03,
            # 仅可见：关闭碰撞，不设置质量；使用 preview surface 材质着色为红色
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
# —— 以下派生类保持原有结构，方便替换到你现有工程中使用 ——


@configclass
class ClawSmplEnvCfg(ClawEnvCfg):
    """使用 SMPL 类人（示例占位），可按需替换为你的爪机/机械臂资产。"""
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
    """使用 Unitree G1（示例占位）"""
    reference_body: str = "torso_link"
    contact_body: str = "right_palm_link"
    key_body_names = ["right_palm_link", "left_palm_link", "right_ankle_roll_link", "left_ankle_roll_link"]
    robot: ArticulationCfg = G1_CFG.replace(prim_path="/World/envs/env_.*/Robot")


@configclass
class ClawH1EnvCfg(ClawEnvCfg):
    """使用 Unitree H1（示例占位）"""
    reference_body: str = "torso_link"
    contact_body: str = "right_elbow_link"
    key_body_names = ["right_elbow_link", "left_elbow_link", "right_ankle_link", "left_ankle_link"]
    robot: ArticulationCfg = H1_CFG.replace(prim_path="/World/envs/env_.*/Robot")
