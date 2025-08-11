# -*- coding: utf-8 -*-
from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import quat_apply

# =======================
# 配置：只保留 PPO 需要的字段
# =======================
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.utils import configclass
from .hoi_cfg import HOIEnvCfg

# =======================
# 任务环境（纯 PPO）
# =======================
class HOIEnv(DirectRLEnv):
    cfg: HOIEnvCfg

    def __init__(self, cfg: HOIEnvCfg, render_mode: str | None = None, **kwargs):
        # 先让基类搭好场景（会调用 _setup_scene），然后才有 robot/obj 的 tensor
        super().__init__(cfg, render_mode, **kwargs)

        # --------- 任务相关索引/常量 ----------
        key_body_names = self.cfg.key_body_names
        self.ref_body_index = self.robot.data.body_names.index(self.cfg.reference_body)
        self.key_body_indexes = [self.robot.data.body_names.index(n) for n in key_body_names]

        # --------- 动态计算 space 维度并刷新 ----------
        ndof = self.robot.data.joint_pos.shape[1]
        num_keys = len(key_body_names)
        # 自身体态：关节 + 根高 + 根姿(6D) + 根线/角速 + 关键点相对位置
        base_self_dim = 2 * ndof + 1 + 6 + 3 + 3 + 3 * num_keys
        # 交互量：椅子相对位置 + 椅子旋转6D + 椅子线/角速 + 相位 + 相位归一化时间
        inter_dim = 3 + 6 + 3 + 3 + 1 + 1
        obs_dim = base_self_dim + inter_dim

        # 写回 cfg 并刷新 Gym spaces（会重建 self.actions）
        self.cfg.action_space = ndof                               # 或者 gym.spaces.Box(low=-1, high=1, shape=(ndof,))
        self.cfg.observation_space = obs_dim                       # 或者 gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,))
        self._configure_gym_env_spaces()

        # --------- 动作缩放（把 [-1,1] 映射到软限） ----------
        dof_lower = self.robot.data.soft_joint_pos_limits[0, :, 0]
        dof_upper = self.robot.data.soft_joint_pos_limits[0, :, 1]
        self.action_offset = 0.5 * (dof_upper + dof_lower)
        self.action_scale = (dof_upper - dof_lower).clamp_min(1e-6)  # 防止零跨度

        # --------- 阶段/计时/缓存 ----------
        self.phase = torch.zeros((self.num_envs,), dtype=torch.long, device=self.device)       # 0: 坐下, 1: 起身
        self.phase_time = torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device)
        self._prev_actions = torch.zeros((self.num_envs, ndof), device=self.device)

    # ----------------- 场景构建 -----------------
    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot)
        self.obj = RigidObject(self.cfg.obj)

        # 地面
        spawn_ground_plane(
            prim_path="/World/ground",
            cfg=GroundPlaneCfg(
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    static_friction=1.0, dynamic_friction=1.0, restitution=0.0
                ),
            ),
        )
        # 克隆并过滤跨 env 碰撞
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=["/World/ground"])

        # 注册到场景
        self.scene.articulations["robot"] = self.robot
        self.scene.rigid_objects["obj"] = self.obj

        # 光照
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    # ----------------- 动作应用 -----------------
    def _pre_physics_step(self, actions: torch.Tensor):
        actions = torch.nan_to_num(actions, nan=0.0, posinf=1.0, neginf=-1.0)
        self.actions = actions.clamp(-1.0, 1.0)

    def _apply_action(self):
        target = self.action_offset + self.action_scale * self.actions
        self.robot.set_joint_position_target(target)

    # ----------------- 观测 -----------------
    def _get_observations(self) -> dict:
        # 自身体态
        obs_self = compute_obs(
            self.robot.data.joint_pos,
            self.robot.data.joint_vel,
            self.robot.data.body_pos_w[:, self.ref_body_index],
            self.robot.data.body_quat_w[:, self.ref_body_index],
            self.robot.data.body_lin_vel_w[:, self.ref_body_index],
            self.robot.data.body_ang_vel_w[:, self.ref_body_index],
            self.robot.data.body_pos_w[:, self.key_body_indexes],
        )

        # 椅子相对量 + 相位
        root_pos = self.robot.data.body_pos_w[:, self.ref_body_index]
        obj_pos = self.obj.data.root_pos_w
        obj_quat = self.obj.data.root_quat_w
        obj_lin = self.obj.data.root_lin_vel_w
        obj_ang = self.obj.data.root_ang_vel_w

        obj_pos_rel = obj_pos - root_pos
        obj_rot_6d = quaternion_to_tangent_and_normal(obj_quat)
        sit_T, stand_T = self.cfg.sit_duration_s, self.cfg.stand_duration_s
        phase01 = self.phase.float().unsqueeze(-1)
        phase_time01 = (
            self.phase_time / torch.where(self.phase == 0,
                                          torch.tensor(sit_T, device=self.device),
                                          torch.tensor(stand_T, device=self.device))
        ).clamp(0, 1).unsqueeze(-1)

        obs_interact = torch.cat([obj_pos_rel, obj_rot_6d, obj_lin, obj_ang, phase01, phase_time01], dim=-1)
        obs = torch.cat([obs_self, obs_interact], dim=-1)
        obs = torch.nan_to_num(obs)  # 全量兜底
        return {"policy": obs}

    # ----------------- 奖励 -----------------
    def _get_rewards(self) -> torch.Tensor:
        cfg = self.cfg
        dt = cfg.sim.dt * cfg.decimation

        root_pos = self.robot.data.body_pos_w[:, self.ref_body_index]
        root_quat = self.robot.data.body_quat_w[:, self.ref_body_index]
        root_lin = self.robot.data.body_lin_vel_w[:, self.ref_body_index]
        root_ang_w = self.robot.data.body_ang_vel_w[:, self.ref_body_index]
        obj_pos = self.obj.data.root_pos_w

        # 距离 / 高度
        seat_xy_err = torch.linalg.norm((root_pos[:, :2] - obj_pos[:, :2]), dim=-1)
        pelvis_h = root_pos[:, 2]
        seat_h = torch.as_tensor(cfg.seat_height, device=self.device)

        # ---------- 目标/朝向/姿态/速度（仿照 locomotion） ----------
        # 目标方向（XY）
        to_goal_xy = obj_pos - root_pos
        to_goal_xy[:, 2] = 0.0
        dist_goal = torch.linalg.norm(to_goal_xy[:, :2], dim=-1).clamp_min(1e-6)
        dir_xy = to_goal_xy[:, :2] / dist_goal.unsqueeze(-1)  # 单位向量

        # 机体前向与竖直（世界系）
        ex = torch.zeros_like(root_pos); ex[:, 0] = 1.0
        ez = torch.zeros_like(root_pos); ez[:, 2] = 1.0
        fwd_w = quat_apply(root_quat, ex)     # 世界系前向
        up_w  = quat_apply(root_quat, ez)     # 世界系竖直
        heading_proj = torch.sum(fwd_w[:, :2] * dir_xy, dim=-1).clamp(-1.0, 1.0)  # cos(朝向误差)
        up_proj = up_w[:, 2].clamp(-1.0, 1.0)

        # 线速度到机体坐标
        q_conj = quat_conjugate_wxyz(root_quat)
        vel_loc = quat_apply(q_conj, root_lin)   # 局部速度
        # 角速度范数（世界系）
        root_ang = torch.linalg.norm(root_ang_w, dim=-1)

        # ----------------- 原有奖励 -----------------
        # 坐下阶段（phase=0）
        r_reach = torch.exp(-3.0 * seat_xy_err)
        r_sit_h = torch.exp(-10.0 * torch.abs(pelvis_h - seat_h))
        r_contact = (torch.abs(pelvis_h - seat_h) < 0.05).float()

        # 起身阶段（phase=1）
        target_stand_h = seat_h + 0.35
        r_stand_h = torch.clamp(pelvis_h - target_stand_h, min=0.0)
        r_leave = (torch.abs(pelvis_h - seat_h) > 0.10).float()

        # 稳定项 + 动作正则（原）
        r_stable = torch.exp(-0.5 * root_ang)
        act_pen = torch.sum(self.actions**2, dim=-1)

        # ----------------- 新增：goal_reached + 稳定行走（仿 locomotion） -----------------
        # 超参数（可在 cfg 里添加同名字段覆盖默认值）
        goal_threshold      = getattr(cfg, "goal_threshold", 0.25)
        goal_bonus          = getattr(cfg, "goal_bonus", 3.0)
        heading_weight_walk = getattr(cfg, "heading_weight_walk", 0.5)
        up_weight_walk      = getattr(cfg, "up_weight_walk", 0.3)
        target_speed        = getattr(cfg, "target_speed", 1.0)
        speed_sigma         = getattr(cfg, "speed_sigma", 0.4)
        speed_weight        = getattr(cfg, "speed_weight", 0.5)
        lateral_vel_weight  = getattr(cfg, "lateral_vel_weight", 0.05)
        vertical_vel_weight = getattr(cfg, "vertical_vel_weight", 0.05)
        angvel_weight       = getattr(cfg, "angvel_weight", 0.05)

        # goal_reached：在阈值内给 bonus（步进式；如需“一次性”，可配合标志位只发一次）
        r_goal_bonus = (dist_goal < goal_threshold).float() * goal_bonus

        # heading 奖励（分段线性，仿 locomotion）
        heading_reward = torch.where(
            heading_proj > 0.8, 
            torch.ones_like(heading_proj) * heading_weight_walk,
            heading_weight_walk * heading_proj / 0.8
        )

        # up 奖励（直立阈值）
        up_reward = torch.where(up_proj > 0.93, torch.ones_like(up_proj) * up_weight_walk, torch.zeros_like(up_proj))

        # 速度跟踪（前向 vx 贴近 target_speed 的高斯项）
        v_forward = vel_loc[:, 0]
        r_speed = speed_weight * torch.exp(-0.5 * ((v_forward - target_speed) / (speed_sigma + 1e-6))**2)

        # 侧/竖直速度惩罚 + 角速度惩罚（稳定行走）
        v_lat_pen  = lateral_vel_weight  * (vel_loc[:, 1] ** 2)
        v_vert_pen = vertical_vel_weight * (vel_loc[:, 2] ** 2)
        ang_pen    = angvel_weight       * (root_ang ** 2)

        r_walk_stability = heading_reward + up_reward + r_speed - v_lat_pen - v_vert_pen - ang_pen

        # 仅在“坐下阶段”接近椅子时鼓励稳定行走
        is_sit = (self.phase == 0)

        reward = (
            is_sit.float() * (cfg.w_reach_seat * r_reach + cfg.w_sit_height * r_sit_h + cfg.w_seat_contact * r_contact + cfg.w_stability * r_stable)
            + (~is_sit).float() * (cfg.w_stand_height * r_stand_h + cfg.w_leave_seat * r_leave + cfg.w_stability * r_stable)
            - cfg.w_action * act_pen
        )

        # 叠加新奖励（坐下阶段）
        reward = reward + is_sit.float() * (r_goal_bonus + r_walk_stability)

        # 相位切换
        self.phase_time += dt
        switch_mask = (self.phase == 0) & (self.phase_time >= cfg.sit_duration_s)
        self.phase[switch_mask] = 1
        self.phase_time[switch_mask] = 0.0
        return reward

    # ----------------- 终止 -----------------
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # time_out = self.episode_length_buf >= self.max_episode_length - 1
        # if self.cfg.early_termination:
        #     died = self.robot.data.body_pos_w[:, self.ref_body_index, 2] < self.cfg.termination_height
        # else:
        #     died = torch.zeros_like(time_out)
        # done_success = (self.phase == 1) & (self.phase_time >= self.cfg.stand_duration_s)
        # return (died | done_success), time_out

        time_out = self.episode_length_buf >= self.max_episode_length - 1
        # 早停条件
        if self.cfg.early_termination:
            died = self.robot.data.body_pos_w[:, self.ref_body_index, 2] < self.cfg.termination_height
        else:
            died = torch.zeros_like(time_out)

        # 新增：非有限值/飞出边界保护
        rp = self.robot.data.body_pos_w[:, self.ref_body_index]
        rq = self.robot.data.body_quat_w[:, self.ref_body_index]
        bad = (~torch.isfinite(rp).all(dim=-1)) | (~torch.isfinite(rq).all(dim=-1))
        # 简单的“飞走”保护（比如位置超过某个范围）
        bad = bad | (rp.abs().max(dim=-1).values > 1e4)

        done_success = (self.phase == 1) & (self.phase_time >= self.cfg.stand_duration_s)
        return (died | done_success | bad), time_out


    # ----------------- 重置 -----------------
    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.robot._ALL_INDICES
        self.robot.reset(env_ids)
        super()._reset_idx(env_ids)

        # 机器人状态
        if self.cfg.reset_strategy == "default":
            root_state, joint_pos, joint_vel = self._reset_strategy_default(env_ids)
        elif self.cfg.reset_strategy == "random":
            root_state, joint_pos, joint_vel = self._reset_strategy_random(env_ids)
        else:
            raise ValueError(f"Unknown reset strategy: {self.cfg.reset_strategy}")

        self.robot.write_root_link_pose_to_sim(root_state[:, :7], env_ids)
        self.robot.write_root_com_velocity_to_sim(root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        # 椅子随机落位（在各自 env 原点附近）
        origins = self.scene.env_origins[env_ids]            # [N, 3]
        rand_xy = (torch.rand((len(env_ids), 2), device=self.device) - 0.5) * 2.0 * self.cfg.chair_xy_spawn_range
        chair_pos = origins.clone()
        chair_pos[:, 0:2] += rand_xy
        yaw = (torch.rand((len(env_ids),), device=self.device) - 0.5) * 2.0 * self.cfg.chair_yaw_spawn_range
        cos, sin = torch.cos(yaw * 0.5), torch.sin(yaw * 0.5)
        chair_quat = torch.stack([cos, 0 * yaw, 0 * yaw, sin], dim=-1)   # 注意 Isaac Lab 四元数是 wxyz
        chair_root_pose = torch.cat([chair_pos, chair_quat], dim=-1)     # (N, 7)
        self.obj.write_root_pose_to_sim(chair_root_pose, env_ids)        # 期望 (N,7)，API 即如此要求

        # 相位清零
        self.phase[env_ids] = 0
        self.phase_time[env_ids] = 0.0
        self._prev_actions[env_ids] = 0.0



    # --- 策略 1：默认重置 ---
    def _reset_strategy_default(self, env_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        root_state = self.robot.data.default_root_state[env_ids].clone()
        root_state[:, :3] += self.scene.env_origins[env_ids]
        joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        joint_vel = self.robot.data.default_joint_vel[env_ids].clone()
        return root_state, joint_pos, joint_vel

    # --- 策略 2：关节随机化（不依赖 motion 数据） ---
    def _reset_strategy_random(self, env_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        root_state, joint_pos, joint_vel = self._reset_strategy_default(env_ids)
        # 在软限内小幅随机
        lo = self.robot.data.soft_joint_pos_limits[0, :, 0]
        hi = self.robot.data.soft_joint_pos_limits[0, :, 1]
        span = (hi - lo).clamp_min(1e-6)
        noise = (torch.rand_like(joint_pos) - 0.5) * 0.2 * span  # ±10% 跨度
        joint_pos = (joint_pos + noise).clamp(lo, hi)
        joint_vel = torch.zeros_like(joint_vel)
        return root_state, joint_pos, joint_vel


# =============== 观测工具函数（与上文 obs_self 对齐） ===============
@torch.jit.script
def quaternion_to_tangent_and_normal(q: torch.Tensor) -> torch.Tensor:
    ref_tangent = torch.zeros_like(q[..., :3]); ref_tangent[..., 0] = 1
    ref_normal = torch.zeros_like(q[..., :3]);  ref_normal[..., -1] = 1
    tangent = quat_apply(q, ref_tangent)
    normal = quat_apply(q, ref_normal)
    return torch.cat([tangent, normal], dim=len(tangent.shape) - 1)

@torch.jit.script
def quat_conjugate_wxyz(q: torch.Tensor) -> torch.Tensor:
    # Isaac Lab 四元数按 wxyz 存放
    return torch.stack((q[..., 0], -q[..., 1], -q[..., 2], -q[..., 3]), dim=-1)

@torch.jit.script
def compute_obs(
    dof_positions: torch.Tensor,
    dof_velocities: torch.Tensor,
    root_positions: torch.Tensor,
    root_rotations: torch.Tensor,
    root_linear_velocities: torch.Tensor,
    root_angular_velocities: torch.Tensor,
    key_body_positions: torch.Tensor,
) -> torch.Tensor:
    obs = torch.cat(
        (
            dof_positions,
            dof_velocities,
            root_positions[:, 2:3],  # 根高度
            quaternion_to_tangent_and_normal(root_rotations),  # 6D
            root_linear_velocities,
            root_angular_velocities,
            (key_body_positions - root_positions.unsqueeze(-2)).view(key_body_positions.shape[0], -1),
        ),
        dim=-1,
    )
    return obs