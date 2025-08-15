# Copyright (c) 2022-2025
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations
import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import quat_apply
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.utils import configclass
from .hoi_cfg import HOIEnvCfg


class HOIEnv(DirectRLEnv):
    cfg: HOIEnvCfg

    def __init__(self, cfg: HOIEnvCfg, render_mode: str | None = None, **kwargs):
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
        frame_dim = base_self_dim + inter_dim  # 单帧观测维度

        # 历史帧堆叠（含当前帧）
        self.history_len = int(getattr(self.cfg, "history_len", 5))
        obs_dim = frame_dim * self.history_len

        self.cfg.action_space = ndof
        self.cfg.observation_space = obs_dim
        self._configure_gym_env_spaces()

        # 历史缓存： [N, H, frame_dim]
        self._frame_dim = frame_dim
        self._obs_hist = torch.zeros((self.num_envs, self.history_len, frame_dim), device=self.device)

        # --------- 动作缩放 ----------
        dof_lower = self.robot.data.soft_joint_pos_limits[0, :, 0]
        dof_upper = self.robot.data.soft_joint_pos_limits[0, :, 1]
        self.action_offset = 0.5 * (dof_upper + dof_lower)
        self.action_scale = (dof_upper - dof_lower).clamp_min(1e-6)

        # --------- 阶段/计时/缓存 ----------
        # phase: 0=坐下(sit)，1=起身/离开(stand->leave)
        self.phase = torch.zeros((self.num_envs,), dtype=torch.long, device=self.device)
        self.phase_time = torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device)
        self.actions = torch.zeros((self.num_envs, ndof), device=self.device)
        self._prev_actions = torch.zeros((self.num_envs, ndof), device=self.device)

        # —— 进度 & 卡住监测 —— #
        self._prev_goal_dist = torch.zeros((self.num_envs,), device=self.device)
        self._no_progress_time = torch.zeros((self.num_envs,), device=self.device)

    # ----------------- 场景构建 -----------------
    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot)
        self.obj = RigidObject(self.cfg.obj)
        spawn_ground_plane(
            prim_path="/World/ground",
            cfg=GroundPlaneCfg(
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    static_friction=1.0, dynamic_friction=1.0, restitution=0.0
                ),
            ),
        )
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=["/World/ground"])
        self.scene.articulations["robot"] = self.robot
        self.scene.rigid_objects["obj"] = self.obj

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    # ----------------- 动作应用 -----------------
    def _pre_physics_step(self, actions: torch.Tensor):
        # 保存上一帧动作（action rate 用）
        self._prev_actions.copy_(self.actions)
        actions = torch.nan_to_num(actions, nan=0.0, posinf=1.0, neginf=-1.0)
        self.actions = actions.clamp(-1.0, 1.0)

    def _apply_action(self):
        target = self.action_offset + self.action_scale * self.actions
        # —— 早期探索噪声：按每回合前 exploration_noise_until_s 秒注入 —— #
        if self.cfg.exploration_noise_std > 0:
            t_sec = self.episode_length_buf.float() * self.cfg.sim.dt * self.cfg.decimation
            mask = (t_sec < self.cfg.exploration_noise_until_s).unsqueeze(-1)
            target = target + mask * (self.cfg.exploration_noise_std * self.action_scale * torch.randn_like(target))
        self.robot.set_joint_position_target(target)

    # ----------------- 观测 -----------------
    def _get_observations(self) -> dict:
        cfg = self.cfg
        obs_self = compute_obs(
            self.robot.data.joint_pos,
            self.robot.data.joint_vel * cfg.dof_vel_scale,   # 速度缩放归一化
            self.robot.data.body_pos_w[:, self.ref_body_index],
            self.robot.data.body_quat_w[:, self.ref_body_index],
            self.robot.data.body_lin_vel_w[:, self.ref_body_index],
            self.robot.data.body_ang_vel_w[:, self.ref_body_index],
            self.robot.data.body_pos_w[:, self.key_body_indexes],
        )

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
            self.phase_time / torch.where(
                self.phase == 0,
                torch.tensor(sit_T, device=self.device),
                torch.tensor(stand_T, device=self.device),
            )
        ).clamp(0, 1).unsqueeze(-1)

        frame = torch.cat([obs_self, obj_pos_rel, obj_rot_6d, obj_lin, obj_ang, phase01, phase_time01], dim=-1)
        frame = torch.nan_to_num(frame)

        # 推入历史缓存并展平
        self._obs_hist = torch.roll(self._obs_hist, shifts=-1, dims=1)
        self._obs_hist[:, -1, :] = frame
        obs = self._obs_hist.reshape(self.num_envs, -1)
        return {"policy": obs}

    # ----------------- 奖励：分阶段主导 -----------------
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

        # 目标方向（XY）
        to_goal_xy = obj_pos - root_pos
        to_goal_xy[:, 2] = 0.0
        dist_goal = torch.linalg.norm(to_goal_xy[:, :2], dim=-1).clamp_min(1e-6)
        dir_xy = to_goal_xy[:, :2] / dist_goal.unsqueeze(-1)

        # 前向/竖直投影 + 局部速度
        ex = torch.zeros_like(root_pos); ex[:, 0] = 1.0
        ez = torch.zeros_like(root_pos); ez[:, 2] = 1.0
        fwd_w = quat_apply(root_quat, ex)
        up_w  = quat_apply(root_quat, ez)
        heading_proj = torch.sum(fwd_w[:, :2] * dir_xy, dim=-1).clamp(-1.0, 1.0)
        up_proj = up_w[:, 2].clamp(-1.0, 1.0)
        q_conj = quat_conjugate_wxyz(root_quat)
        vel_loc = quat_apply(q_conj, root_lin)
        root_ang = torch.linalg.norm(root_ang_w, dim=-1)

        # ---------- 原子奖励项 ----------
        # 坐下阶段基础
        r_reach   = torch.exp(-3.0 * seat_xy_err)
        r_sit_h   = torch.exp(-10.0 * torch.abs(pelvis_h - seat_h))
        r_contact = (torch.abs(pelvis_h - seat_h) < 0.05).float()  # 高度代理（可换接触）

        # 起身/离开阶段基础
        target_stand_h = seat_h + cfg.stand_height_offset
        r_stand_h = torch.clamp(pelvis_h - target_stand_h, min=0.0)
        r_leave   = (torch.abs(pelvis_h - seat_h) > 0.10).float()

        # 稳定与动作正则（部分留到统一代价处）
        r_stable = torch.exp(-0.5 * root_ang)

        # 行走控制
        heading_reward = torch.where(
            heading_proj > 0.8, torch.ones_like(heading_proj) * cfg.heading_weight_walk,
            cfg.heading_weight_walk * heading_proj / 0.8,
        )
        up_reward = torch.where(up_proj > 0.93, torch.ones_like(up_proj) * cfg.up_weight_walk, torch.zeros_like(up_proj))
        v_forward = vel_loc[:, 0]
        r_speed = cfg.speed_weight * torch.exp(-0.5 * ((v_forward - cfg.target_speed) / (cfg.speed_sigma + 1e-6))**2)
        v_lat_pen  = cfg.lateral_vel_weight  * (vel_loc[:, 1] ** 2)
        v_vert_pen = cfg.vertical_vel_weight * (vel_loc[:, 2] ** 2)
        ang_pen    = cfg.angvel_weight       * (root_ang ** 2)
        r_walk_ctrl = heading_reward + up_reward + r_speed - v_lat_pen - v_vert_pen - ang_pen

        # 站立姿态奖励（靠近阶段）
        r_stand_posture_h = torch.exp(-0.5 * ((pelvis_h - target_stand_h) / (cfg.stand_posture_sigma + 1e-6))**2)
        r_stand_posture = r_stand_posture_h + 0.5 * up_reward

        # 指定区域奖励（靠近区）
        r_enter_zone_bonus = (dist_goal < cfg.approach_zone_radius).float() * cfg.enter_zone_bonus

        # 目标达成奖励（接近目标阈值）
        r_goal_bonus = (dist_goal < cfg.goal_threshold).float() * cfg.goal_bonus

        # 远离奖励（离开阶段）
        r_away = 1.0 - torch.exp(-cfg.away_beta * dist_goal)

        # ---------- 平滑门控 ----------
        g_approach = torch.sigmoid(cfg.blend_k * (dist_goal - cfg.approach_dist_th))
        g_leave = torch.sigmoid(cfg.blend_k * (dist_goal - cfg.leave_dist_th))

        # ---------- 分块组合 ----------
        block_approach = cfg.w_reach_seat * r_reach + r_walk_ctrl + cfg.w_stand_posture * r_stand_posture + r_enter_zone_bonus
        block_sit      = cfg.w_sit_height * r_sit_h + cfg.w_seat_contact * r_contact

        reward_phase0 = (
            cfg.w_block_approach * g_approach * block_approach
            + cfg.w_block_sit    * (1.0 - g_approach) * block_sit
            + cfg.w_stability * r_stable
            + r_goal_bonus
        )

        block_stand = cfg.w_stand_height * r_stand_h
        block_leave = cfg.w_leave_seat   * r_leave + r_away

        reward_phase1 = (
            cfg.w_block_stand * (1.0 - g_leave) * block_stand
            + cfg.w_block_leave * g_leave * block_leave
            + cfg.w_stability * r_stable
        )

        is_sit = (self.phase == 0)
        reward = torch.where(is_sit, reward_phase0, reward_phase1)

        # ====== 强化探索与站立：进度 + upright 门控 alive + 统一代价 ======
        # 进度奖励：上一步距离 - 当前距离（>0 表示更近）
        progress = (self._prev_goal_dist - dist_goal).clamp_(-0.5, 0.5)
        reward = reward + cfg.progress_weight * progress

        # upright 门控 alive：不直立不给或打折
        alive_gain = (up_proj - cfg.alive_upright_min) / (1.0 - cfg.alive_upright_min + 1e-6)
        alive_gain = torch.clamp(alive_gain, 0.0, 1.0)
        reward = reward + cfg.alive_reward_scale * alive_gain

        # 能耗/动作幅度/变化率
        efforts = None
        try:
            eff_list = []
            for grp in getattr(self.robot, "actuators", {}).values():
                if hasattr(grp, "applied_effort") and grp.applied_effort is not None:
                    eff_list.append(grp.applied_effort)
            if len(eff_list) > 0:
                efforts = torch.cat(eff_list, dim=-1)
        except Exception:
            efforts = None
        if efforts is None or efforts.shape[-1] != self.robot.data.joint_vel.shape[-1]:
            efforts = torch.abs(self.actions)

        energy_pen = cfg.energy_cost_scale * torch.sum(torch.abs(efforts) * torch.abs(self.robot.data.joint_vel), dim=-1)
        act_pen    = torch.sum(self.actions**2, dim=-1)
        act_rate_pen = torch.sum((self.actions - self._prev_actions)**2, dim=-1)

        reward = reward - energy_pen - cfg.actions_cost_scale * act_pen - cfg.w_action_rate * act_rate_pen

        # —— 更新 prev_dist & 卡住计时 —— 
        self._no_progress_time += torch.where(
            (progress <= 1e-3) & (dist_goal > cfg.goal_threshold),
            torch.full_like(progress, fill_value=dt),
            torch.zeros_like(progress),
        )
        self._prev_goal_dist = dist_goal.detach()

        # 相位切换（仍按时间；“进入区”通过奖励驱动）
        self.phase_time += dt
        switch_mask = (self.phase == 0) & (self.phase_time >= cfg.sit_duration_s)
        self.phase[switch_mask] = 1
        self.phase_time[switch_mask] = 0.0
        return reward

    # ----------------- 终止 -----------------
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        rp = self.robot.data.body_pos_w[:, self.ref_body_index]
        rq = self.robot.data.body_quat_w[:, self.ref_body_index]
        bad = (~torch.isfinite(rp).all(dim=-1)) | (~torch.isfinite(rq).all(dim=-1))
        bad = bad | (rp.abs().max(dim=-1).values > 1e4)

        # 高度 + 倾斜（upright）
        ez = torch.zeros_like(rp); ez[:, 2] = 1.0
        up_w = quat_apply(rq, ez)
        up_proj = up_w[:, 2]
        fell = (rp[:, 2] < self.cfg.termination_height) | (up_proj < self.cfg.tilt_termination_cos)

        # 卡住（长时间无“朝目标”进展）
        stuck = self._no_progress_time > self.cfg.stuck_time_s

        done_success = (self.phase == 1) & (self.phase_time >= self.cfg.stand_duration_s)
        return (fell | done_success | bad | stuck), time_out

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

        # 椅子随机落位（课程：范围较小，学会后再增大到 3.0 / 0.5）
        origins = self.scene.env_origins[env_ids]
        rand_xy = (torch.rand((len(env_ids), 2), device=self.device) - 0.5) * 2.0 * self.cfg.chair_xy_spawn_range
        chair_pos = origins.clone()
        chair_pos[:, 0:2] += rand_xy
        yaw = (torch.rand((len(env_ids),), device=self.device) - 0.5) * 2.0 * self.cfg.chair_yaw_spawn_range
        cos, sin = torch.cos(yaw * 0.5), torch.sin(yaw * 0.5)
        chair_quat = torch.stack([cos, 0 * yaw, 0 * yaw, sin], dim=-1)  # wxyz
        chair_root_pose = torch.cat([chair_pos, chair_quat], dim=-1)
        self.obj.write_root_pose_to_sim(chair_root_pose, env_ids)

        # 相位/缓存清零
        self.phase[env_ids] = 0
        self.phase_time[env_ids] = 0.0
        self.actions[env_ids] = 0.0
        self._prev_actions[env_ids] = 0.0
        self._obs_hist[env_ids] = 0.0

        # —— 初始化“上一步目标距离”和“无进展计时” —— #
        # 用我们刚写入的 root_state 与 chair_pos 计算
        root_pos0 = root_state[:, :3]
        dist0 = torch.linalg.norm((chair_pos[:, :2] - root_pos0[:, :2]), dim=-1)
        self._prev_goal_dist[env_ids] = dist0
        self._no_progress_time[env_ids] = 0.0

    def _reset_strategy_default(self, env_ids: torch.Tensor):
        root_state = self.robot.data.default_root_state[env_ids].clone()
        root_state[:, :3] += self.scene.env_origins[env_ids]
        joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        joint_vel = self.robot.data.default_joint_vel[env_ids].clone()
        return root_state, joint_pos, joint_vel

    def _reset_strategy_random(self, env_ids: torch.Tensor):
        root_state, joint_pos, joint_vel = self._reset_strategy_default(env_ids)
        lo = self.robot.data.soft_joint_pos_limits[0, :, 0]
        hi = self.robot.data.soft_joint_pos_limits[0, :, 1]
        span = (hi - lo).clamp_min(1e-6)
        noise = (torch.rand_like(joint_pos) - 0.5) * 0.2 * span
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
            dof_velocities,  # 已在调用处做了缩放
            root_positions[:, 2:3],
            quaternion_to_tangent_and_normal(root_rotations),
            root_linear_velocities,
            root_angular_velocities,
            (key_body_positions - root_positions.unsqueeze(-2)).view(key_body_positions.shape[0], -1),
        ),
        dim=-1,
    )
    return obs
