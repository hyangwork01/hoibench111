# Copyright (c) 2022-2025
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations
import torch
import torch.nn.functional as F
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import quat_apply
from isaaclab.envs.common import VecEnvObs, VecEnvStepReturn
from collections.abc import Sequence
from math import pi

from .base import HOIEnv
from .touch_cfg import TouchEnvCfg


class TouchEnv(HOIEnv):
    cfg: TouchEnvCfg

    def __init__(self, cfg: TouchEnvCfg, render_mode: str | None = None, **kwargs):
        self.env_spacing = cfg.scene.env_spacing
        self.env_sample_len = self.env_spacing - 2

        super().__init__(cfg, render_mode, **kwargs)

        ndof = self.robot.data.joint_pos.shape[1]
        base_self_dim = 2 * ndof + 1 + 6 + 3 + 3   # q, qd, root_z, root_rot6d, root_lin, root_ang
        goal_dim = 3
        obs_dim = base_self_dim + goal_dim

        self.cfg.action_space = ndof
        self.cfg.observation_space = obs_dim
        self._configure_gym_env_spaces()

        # action scale/offset
        dof_lower = self.robot.data.soft_joint_pos_limits[0, :, 0]
        dof_upper = self.robot.data.soft_joint_pos_limits[0, :, 1]
        self.action_offset = 0.5 * (dof_upper + dof_lower)
        self.action_scale = (dof_upper - dof_lower).clamp_min(1e-6)

        # stats
        self.total_time = 0.0
        self.total_completed = 0

        # 历史缓存：进度 shaping（-1 表示未初始化）
        self._prev_xy_dist = torch.full((self.num_envs,), -1.0, device=self.device)

    # ----------------- 场景构建 -----------------
    def _setup_scene(self):
        # 机器人
        self.robot = Articulation(self.cfg.robot)

        # 目标小球（仅可见、不可碰撞、运动学体）
        self.goal = RigidObject(self.cfg.goal)

        # 地面
        spawn_ground_plane(
            prim_path="/World/ground",
            cfg=GroundPlaneCfg(
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    static_friction=1.0, dynamic_friction=1.0, restitution=0.0
                ),
            ),
        )

        # 克隆 envs + 过滤跨 env 碰撞
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=["/World/ground"])

        # 注册
        self.scene.articulations["robot"] = self.robot
        self.scene.rigid_objects["goal"] = self.goal

        # 光照
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    # ----------------- 重置（训练：逐 env） -----------------
    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, dtype=torch.int32, device=self.device)
        else:
            env_ids = torch.as_tensor(env_ids, dtype=torch.int32, device=self.device).contiguous()

        if len(env_ids) == 0:
            return
        self.episode_length_buf[env_ids] = 0

        device = self.device
        half_x = half_y = 0.5 * float(self.env_sample_len)
        robot_r = 1.0
        max_trials = 1000
        yaw_range = pi

        # 默认状态（局部坐标）
        root_state = self.robot.data.default_root_state[env_ids].clone()  # (N,13)
        joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        joint_vel = self.robot.data.default_joint_vel[env_ids].clone()

        origins = self.scene.env_origins[env_ids]  # (N,3)

        N = len(env_ids)
        robot_xy = torch.zeros((N, 2), device=device)

        def _sample_xy(cx: float, cy: float, r_keepout: float):
            x_lo, x_hi = cx - (half_x - r_keepout), cx + (half_x - r_keepout)
            y_lo, y_hi = cy - (half_y - r_keepout), cy + (half_y - r_keepout)
            if x_hi < x_lo or y_hi < y_lo:
                return cx, cy
            rx = torch.empty((), device=device).uniform_(x_lo, x_hi).item()
            ry = torch.empty((), device=device).uniform_(y_lo, y_hi).item()
            return rx, ry

        # 放置机器人根位置 + 随机 yaw
        for i in range(N):
            cx, cy, cz = origins[i].tolist()
            placed = False
            for _ in range(max_trials):
                rx, ry = _sample_xy(cx, cy, robot_r)
                robot_xy[i, 0], robot_xy[i, 1] = rx, ry
                placed = True
                break

            world_z = cz + (root_state[i, 2].item() if root_state.ndim == 2 else 0.0)
            root_state[i, 0] = robot_xy[i, 0]
            root_state[i, 1] = robot_xy[i, 1]
            root_state[i, 2] = world_z

        # 写回：根姿态/速度/关节
        yaw = torch.empty((N,), device=device).uniform_(-yaw_range, yaw_range)
        cos, sin = torch.cos(0.5 * yaw), torch.sin(0.5 * yaw)
        quat = torch.stack([cos, torch.zeros_like(cos), torch.zeros_like(cos), sin], dim=-1)  # wxyz
        self.robot.write_root_link_pose_to_sim(
            torch.cat([root_state[:, :3], quat], dim=-1), env_ids
        )
        self.robot.write_root_com_velocity_to_sim(root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        # === 目标点采样 & 放置红球 ===
        if not hasattr(self, "goal_pos_w"):
            self.goal_pos_w = torch.zeros((self.num_envs, 3), device=device)

        goal_pose = torch.zeros((N, 7), device=device)
        goal_pose[:, 3] = 1.0  # 单位四元数（wxyz）

        # 近似可达半径（可在 cfg 中添加 touch_threshold/approx_reach，默认 1.0m）
        approx_R = float(getattr(self.cfg, "approx_reach", 1.0))
        approx_R = max(0.2, approx_R)

        for i in range(N):
            cx, cy, cz = origins[i].tolist()
            # 以 root 为中心采样（无需 sim.forward）
            rx, ry, rz = root_state[i, 0].item(), root_state[i, 1].item(), root_state[i, 2].item()
            # 高度：地面上方 5cm ~ root 上方 approx_R
            z_lo = cz + 0.05
            z_hi = rz + approx_R
            if z_hi <= z_lo:
                z_hi = z_lo + 0.10
            gz = torch.empty((), device=device).uniform_(z_lo, z_hi).item()
            # 平面半径/角度
            r = torch.empty((), device=device).uniform_(0.05, approx_R).item()
            ang = torch.empty((), device=device).uniform_(-pi, pi).item()
            gx = rx + r * torch.cos(torch.tensor(ang, device=device)).item()
            gy = ry + r * torch.sin(torch.tensor(ang, device=device)).item()
            # 环境边界裁剪
            x_lo, x_hi = cx - (0.5 * self.env_sample_len - 0.2), cx + (0.5 * self.env_sample_len - 0.2)
            y_lo, y_hi = cy - (0.5 * self.env_sample_len - 0.2), cy + (0.5 * self.env_sample_len - 0.2)
            gx = min(max(gx, x_lo), x_hi)
            gy = min(max(gy, y_lo), y_hi)

            env_i = int(env_ids[i])
            self.goal_pos_w[env_i] = torch.tensor([gx, gy, gz], device=device)
            goal_pose[i, 0:3] = self.goal_pos_w[env_i]

        self.goal.write_root_pose_to_sim(goal_pose, env_ids)

        # 清计数掩码 + 清上帧观测缓存（仅重置到这些 env）
        if hasattr(self, "_counted_mask"):
            self._counted_mask[env_ids] = False
        if hasattr(self, "_last_obs"):
            for k in self._last_obs:
                self._last_obs[k][env_ids] = 0.0

        # 重置这些 env 的进度缓存
        if hasattr(self, "_prev_xy_dist"):
            self._prev_xy_dist[env_ids] = -1.0

    # === 完成与超时判定 ===
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        device = self.device

        # 超时（truncation）
        time_out = (self.episode_length_buf >= self.max_episode_length - 1).to(device)

        # —— 成功判定：接触体到目标的 3D 距离 ——
        thr = float(getattr(self.cfg, "touch_threshold", 0.03))  # 默认 3 cm
        min_success_steps = int(getattr(self.cfg, "min_success_steps", 5))
        allow_success = (self.episode_length_buf >= min_success_steps)

        names = self.robot.data.body_names
        contact_idx = names.index(self.cfg.contact_body)
        contact_pos = self.robot.data.body_link_pos_w[:, contact_idx]  # (N,3)

        d = torch.linalg.norm(contact_pos - self.goal_pos_w, dim=-1)  # (N,)
        done_success = (d < thr) & allow_success

        # 成功优先（不要同时标记超时）
        time_out = time_out & (~done_success)

        # 统计与边沿触发
        completed_now = done_success | time_out
        if not hasattr(self, "_counted_mask"):
            self._counted_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=device)
            self.stat_success_count = 0
            self.stat_timeout_count = 0
            self.stat_success_time_sum = 0.0
            self.stat_timeout_time_sum = 0.0
            self.stat_completed = 0
            self.stat_avg_time = 0.0

        new_mask = completed_now & (~self._counted_mask)
        new_succ = done_success & new_mask
        new_to = time_out & new_mask

        dt = float(self.cfg.sim.dt) * int(self.cfg.decimation)
        step_times = (self.episode_length_buf.to(torch.float32) + 1.0) * dt

        self.stat_success_count += int(new_succ.sum().item())
        self.stat_timeout_count += int(new_to.sum().item())
        if new_succ.any():
            self.stat_success_time_sum += float(step_times[new_succ].sum().item())
        if new_to.any():
            self.stat_timeout_time_sum += float(step_times[new_to].sum().item())
        self.stat_completed = self.stat_success_count + self.stat_timeout_count
        if self.stat_completed > 0:
            total_time = self.stat_success_time_sum + self.stat_timeout_time_sum
            self.stat_avg_time = total_time / self.stat_completed

        self._counted_mask |= completed_now
        return done_success, time_out

    def step(self, action: torch.Tensor) -> VecEnvStepReturn:
        action = action.to(self.device)
        if self.cfg.action_noise_model:
            action = self._action_noise_model(action)

        # 父类预处理（裁剪/缓存等）
        self._pre_physics_step(action)

        is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()

        for _ in range(self.cfg.decimation):
            self._sim_step_counter += 1
            self._apply_action()
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            if self._sim_step_counter % self.cfg.sim.render_interval == 0 and is_rendering:
                self.sim.render()
            self.scene.update(dt=self.physics_dt)

        # 计数
        self.episode_length_buf += 1
        self.common_step_counter += 1

        # dones / rewards
        self.reset_terminated[:], self.reset_time_outs[:] = self._get_dones()
        self.reset_buf = self.reset_terminated | self.reset_time_outs
        self.reward_buf = self._get_rewards()

        # —— 训练：逐 env 重置（重要）——
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self._reset_idx(reset_env_ids)
            self.scene.write_data_to_sim()
            self.sim.forward()
            if self.sim.has_rtx_sensors() and self.cfg.rerender_on_reset:
                self.sim.render()

        # 事件
        if self.cfg.events:
            if "interval" in self.event_manager.available_modes:
                self.event_manager.apply(mode="interval", dt=self.step_dt)

        # 观测
        self.obs_buf = self._get_observations()

        # 观测噪声（不加到 critic state）
        if self.cfg.observation_noise_model:
            self.obs_buf["policy"] = self._observation_noise_model(self.obs_buf["policy"])

        return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras

    # 训练版：不冻结已完成 env 的动作（保留简单位置控制）
    def _apply_action(self):
        target = self.action_offset + self.action_scale * self.actions
        self.robot.set_joint_position_target(target)

    # 训练版观测：按你在 __init__ 中的 obs 定义
    def _get_observations(self) -> VecEnvObs:
        device = self.device
        ndof = self.robot.data.joint_pos.shape[1]
        q = self.robot.data.joint_pos
        qd = self.robot.data.joint_vel

        # 参考刚体：优先 cfg.reference_body -> root_link
        ref_name = getattr(self.cfg, "reference_body", None)
        names = getattr(self.robot.data, "body_names", self.robot.data.body_names)
        if ref_name and (ref_name in names):
            rb = names.index(ref_name)
            root_pos_w = self.robot.data.body_link_pos_w[:, rb]
            root_quat_w = self.robot.data.body_link_quat_w[:, rb]
            root_lin_w = self.robot.data.body_link_lin_vel_w[:, rb]
            root_ang_w = self.robot.data.body_link_ang_vel_w[:, rb]
        else:
            root_pos_w = self.robot.data.root_link_pos_w
            root_quat_w = self.robot.data.root_link_quat_w
            root_lin_w = self.robot.data.root_link_lin_vel_w
            root_ang_w = self.robot.data.root_link_ang_vel_w

        ex = torch.zeros_like(root_pos_w); ex[:, 0] = 1.0
        ez = torch.zeros_like(root_pos_w); ez[:, 2] = 1.0
        tangent = quat_apply(root_quat_w, ex)
        normal = quat_apply(root_quat_w, ez)
        root_rot_6d = torch.cat([tangent, normal], dim=-1)  # 6D 连续姿态
        root_z = root_pos_w[:, 2:3]

        # 目标相对位移（相对于参考刚体）
        goal_rel = self.goal_pos_w - root_pos_w  # (N,3)

        self_part = torch.cat([q, qd, root_z, root_rot_6d, root_lin_w, root_ang_w], dim=-1)
        policy_obs = torch.cat([self_part, goal_rel], dim=-1)

        # 断言观测维度
        if isinstance(self.cfg.observation_space, int):
            assert policy_obs.shape[1] == self.cfg.observation_space, \
                f"Obs length mismatch: got {policy_obs.shape[1]}, expect {self.cfg.observation_space}"

        obs = {"policy": torch.nan_to_num(policy_obs)}
        self._last_obs = {k: v.clone() for k, v in obs.items()}
        return obs

    # 预物理步处理（健壮化 + 裁剪）
    def _pre_physics_step(self, actions: torch.Tensor):
        self.actions = actions
        self.actions = torch.nan_to_num(self.actions, nan=0.0, posinf=0.0, neginf=0.0)
        self.actions.clamp_(-1.0, 1.0)

    # === 奖励函数（touch） ===
    def _get_rewards(self) -> torch.Tensor:
        """
        组合奖励（围绕“接触体→目标点”的 3D 距离）：
          + 进度 r_progress = prev_dist - cur_dist
          + 距离 r_dist = exp(-dist / sigma)
          + 稳定 r_stable：接近目标时抑制参考刚体线/角速度
          - 正则：动作 / 关节速度 / 接近软限位
          + 事件：成功 +10，超时 -1
          - step_pen：每步 -0.01
        """
        device = self.device
        eps = 1e-6

        # ---------- weights ----------
        w_progress      = 2.0
        w_dist          = 2.0
        w_stable        = 0.5

        w_action        = 2.5e-3
        w_qd            = 1.0e-4
        w_limits        = 2.0e-2

        bonus_success   = 10.0
        penalty_timeout = 1.0
        step_time_pen   = 0.01

        # ---------- 提取状态 ----------
        # 参考刚体（与观测一致）
        ref_name = getattr(self.cfg, "reference_body", None)
        names = self.robot.data.body_names
        if ref_name and (ref_name in names):
            rb = names.index(ref_name)
            root_pos_w  = self.robot.data.body_link_pos_w[:, rb]
            root_quat_w = self.robot.data.body_link_quat_w[:, rb]
            root_lin_w  = self.robot.data.body_link_lin_vel_w[:, rb]
            root_ang_w  = self.robot.data.body_link_ang_vel_w[:, rb]
        else:
            root_pos_w  = self.robot.data.root_link_pos_w
            root_quat_w = self.robot.data.root_link_quat_w
            root_lin_w  = self.robot.data.root_link_lin_vel_w
            root_ang_w  = self.robot.data.root_link_ang_vel_w

        contact_idx = names.index(self.cfg.contact_body)
        contact_pos = self.robot.data.body_link_pos_w[:, contact_idx]

        # 距离到目标
        dist = torch.linalg.norm(contact_pos - self.goal_pos_w, dim=-1)  # (N,)

        # ---------- 各项奖励 ----------
        # 势能进度：prev - cur（首次步用当前距离初始化）
        prev = torch.where(self._prev_xy_dist < 0.0, dist.detach(), self._prev_xy_dist)
        r_progress = w_progress * (prev - dist)
        self._prev_xy_dist = dist.detach()

        # 距离 RBF：越近越接近 1
        sigma = 0.30
        r_dist = w_dist * torch.exp(-dist / (sigma + eps))

        # 近目标稳定（<0.15m 时抑制参考刚体速度）
        near_mask = (dist < 0.15).float()
        r_stable = w_stable * near_mask * (
            1.0 / (1.0 + root_lin_w.norm(dim=-1)) + 1.0 / (1.0 + root_ang_w.norm(dim=-1))
        )

        # 正则化惩罚
        p_action = w_action * (self.actions ** 2).sum(dim=-1)
        qd = self.robot.data.joint_vel
        p_qd = w_qd * (qd ** 2).sum(dim=-1)

        # 接近软限位惩罚（95% 以后）
        lower = self.robot.data.soft_joint_pos_limits[0, :, 0]
        upper = self.robot.data.soft_joint_pos_limits[0, :, 1]
        q     = self.robot.data.joint_pos
        rel   = (q - lower) / (upper - lower + eps)
        margin = torch.clamp(0.95 - torch.minimum(rel, 1 - rel), min=0.0)
        p_limits = w_limits * (margin ** 2).sum(dim=-1)

        # 事件项
        bonus = torch.zeros_like(dist)
        if hasattr(self, "reset_terminated"):
            bonus = bonus + bonus_success * self.reset_terminated.float()
        if hasattr(self, "reset_time_outs"):
            bonus = bonus - penalty_timeout * self.reset_time_outs.float()

        # 每步小负激励
        step_pen = step_time_pen * torch.ones_like(dist)

        reward = (
            r_progress + r_dist + r_stable
            - p_action - p_qd - p_limits
            + bonus - step_pen
        )
        reward = torch.nan_to_num(reward, nan=0.0, posinf=0.0, neginf=0.0)

        # 记录项均值，便于 logger
        self.extras["r_terms"] = {
            "progress": r_progress.mean().item(),
            "dist":     r_dist.mean().item(),
            "stable":   r_stable.mean().item(),
            "p_action": p_action.mean().item(),
            "p_qd":     p_qd.mean().item(),
            "p_limits": p_limits.mean().item(),
            "bonus":    bonus.mean().item(),
        }
        return reward
