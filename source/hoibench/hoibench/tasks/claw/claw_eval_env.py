# Copyright (c) 2022-2025
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations
from math import pi
from collections.abc import Sequence
import torch
import torch.nn.functional as F

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs.common import VecEnvObs, VecEnvStepReturn
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import quat_apply

from .base import HOIEnv
from .claw_cfg import ClawEnvCfg


class ClawEnv(HOIEnv):
    """Claw 评测环境（Eval）：
    - 桌面上随机初始化玩具与目标（目标为可视化小球）；
    - 成功：仅看玩具到目标的 XY 距离 + 连续保持 eval_hold_success_steps 帧；
    - 不做逐 env 重置，所有 env 完成后统一重置；
    - 奖励恒为 0。
    """
    cfg: ClawEnvCfg

    def __init__(self, cfg: ClawEnvCfg, render_mode: str | None = None, **kwargs):
        self.env_spacing = cfg.scene.env_spacing
        self.env_sample_len = self.env_spacing - 2

        super().__init__(cfg, render_mode, **kwargs)

        # === 观测维度（与训练保持一致）===
        ndof = self.robot.data.joint_pos.shape[1]
        base_self_dim = 2 * ndof + 1 + 6 + 3 + 3   # q, qd, root_z, root_rot6d, root_lin, root_ang
        inter_dim = 10                              # toy_center_rel(3)+size_obb(3)+toy_quat(4)
        goal_dim = 2                                # goal_vec_xy
        obs_dim = base_self_dim + inter_dim + goal_dim

        self.cfg.action_space = ndof
        self.cfg.observation_space = obs_dim
        self._configure_gym_env_spaces()

        # 动作缩放
        dof_lower = self.robot.data.soft_joint_pos_limits[0, :, 0]
        dof_upper = self.robot.data.soft_joint_pos_limits[0, :, 1]
        self.action_offset = 0.5 * (dof_upper + dof_lower)
        self.action_scale = (dof_upper - dof_lower).clamp_min(1e-6)

        # 评测设置
        self._thr_xy = float(getattr(self.cfg, "goal_success_xy", 0.03))     # 成功半径（XY），默认 3cm
        self._eval_hold_steps = int(getattr(self.cfg, "eval_hold_success_steps", 5))  # 连续保持成功帧数

        # 目标点（世界系 XY）
        self._goal_xy_w = torch.zeros((self.num_envs, 2), device=self.device)

        # 连续成功计数（逐 env）
        self._succ_streak = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)

        # 统计/标志
        self.stat_success_count = 0
        self.stat_timeout_count = 0
        self.stat_success_time_sum = 0.0
        self.stat_timeout_time_sum = 0.0
        self.stat_completed = 0
        self.stat_avg_time = 0.0
        self._counted_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.done_flag = False  # 整体一次评测是否已完成（全部 env 完成后置 True）

    # ----------------- 场景构建 -----------------
    def _setup_scene(self):
        # 机器人与物体
        self.robot = Articulation(self.cfg.robot)
        self.table = RigidObject(self.cfg.table)
        self.toy   = RigidObject(self.cfg.toy)
        self.goal  = RigidObject(self.cfg.goal)  # 仅可视化：kinematic & no-collision

        # 地面 & 场景克隆
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

        # 注册
        self.scene.articulations["robot"] = self.robot
        self.scene.rigid_objects["table"] = self.table
        self.scene.rigid_objects["toy"]   = self.toy
        self.scene.rigid_objects["goal"]  = self.goal

        # 灯光
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    # ----------------- AABB/OBB 工具（与训练一致） -----------------
    def _get_dims_for(self, obj: RigidObject, env_ids: Sequence[int] | None = None):
        """返回 obj 在各 env 的 AABB/OBB 尺寸信息。"""
        import numpy as np
        import isaacsim.core.utils.bounds as bounds_utils
        import isaacsim.core.utils.stage as stage_utils

        device = self.device
        if env_ids is None:
            env_ids = list(range(self.num_envs))
        else:
            env_ids = list(env_ids) or [0]

        pat = obj.cfg.prim_path
        if "env_.*/" in pat:
            suffix = pat.split("env_.*/", 1)[1]
        elif "{ENV_REGEX_NS}/" in pat:
            suffix = pat.split("{ENV_REGEX_NS}/", 1)[1]
        else:
            if hasattr(self.scene, "env_ns") and f"{self.scene.env_ns}/env_0/" in pat:
                suffix = pat.split(f"{self.scene.env_ns}/env_0/", 1)[1]
            elif "/World/envs/env_0/" in pat:
                suffix = pat.split("/World/envs/env_0/", 1)[1]
            elif pat.startswith("/World/"):
                suffix = pat.split("/World/", 1)[1]
            else:
                suffix = pat
        env_ns = getattr(self.scene, "env_ns", "/World/envs")
        prim_paths = [f"{env_ns}/env_{int(i)}/{suffix.lstrip('/')}" for i in env_ids]

        stage = stage_utils.get_current_stage()
        cache = bounds_utils.create_bbox_cache()

        aabb_min_list, aabb_max_list = [], []
        size_aabb_list, size_obb_list = [], []
        keepout_list = []

        for p in prim_paths:
            prim = stage.GetPrimAtPath(p)
            if not prim.IsValid():
                raise RuntimeError(f"Invalid prim path: {p}")
            try:
                bounds_utils.recompute_extents(prim, include_children=True)
            except Exception:
                pass
            aabb = bounds_utils.compute_aabb(cache, prim_path=p, include_children=True)
            aabb = np.asarray(aabb, dtype=np.float32)
            a_min = torch.tensor(aabb[:3], device=device)
            a_max = torch.tensor(aabb[3:], device=device)
            a_size = a_max - a_min

            _, _, half_extent = bounds_utils.compute_obb(cache, prim_path=p)
            size_obb = torch.tensor(2.0 * np.asarray(half_extent, dtype=np.float32), device=device)

            r_xy = float(0.5 * torch.sqrt(size_obb[0] ** 2 + size_obb[1] ** 2).item())

            aabb_min_list.append(a_min)
            aabb_max_list.append(a_max)
            size_aabb_list.append(a_size)
            size_obb_list.append(size_obb)
            keepout_list.append(r_xy)

        aabb_min = torch.stack(aabb_min_list, dim=0)
        aabb_max = torch.stack(aabb_max_list, dim=0)
        size_aabb = torch.stack(size_aabb_list, dim=0)
        size_obb = torch.stack(size_obb_list, dim=0)
        keepout_radius_xy = torch.tensor(keepout_list, device=device)

        return {
            "size_obb": size_obb,
            "size_aabb": size_aabb,
            "aabb_min": aabb_min,
            "aabb_max": aabb_max,
            "keepout_radius_xy": keepout_radius_xy,
        }

    # ----------------- 重置（Eval：仅在需要的 env 上采样） -----------------
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

        # 尺寸
        table_dims = self._get_dims_for(self.table, env_ids)
        toy_dims   = self._get_dims_for(self.toy,   env_ids)

        # 默认状态
        root_state  = self.robot.data.default_root_state[env_ids].clone()
        joint_pos   = self.robot.data.default_joint_pos[env_ids].clone()
        joint_vel   = self.robot.data.default_joint_vel[env_ids].clone()
        table_state = self.table.data.default_root_state[env_ids].clone()
        toy_state   = self.toy.data.default_root_state[env_ids].clone()

        origins = self.scene.env_origins[env_ids]

        N = len(env_ids)
        robot_xy = torch.zeros((N, 2), device=device)
        toy_xy   = torch.zeros((N, 2), device=device)
        toy_yaw  = torch.zeros((N,),   device=device)
        goal_xy  = torch.zeros((N, 2), device=device)
        use_default_quat_toy = torch.zeros((N,), dtype=torch.bool, device=device)

        # 矩形内均匀采样
        def _sample_xy_box(cx: float, cy: float, half_w: float, half_d: float):
            rx = torch.empty((), device=device).uniform_(cx - half_w, cx + half_w).item()
            ry = torch.empty((), device=device).uniform_(cy - half_d, cy + half_d).item()
            return rx, ry

        for i in range(N):
            cx, cy, cz = origins[i].tolist()

            # --- robot 放置：与桌子保持距离 ---
            r_table = float(table_dims["keepout_radius_xy"][i].item())
            placed_robot = False
            for _ in range(max_trials):
                rx = torch.empty((), device=device).uniform_(cx - (half_x - robot_r), cx + (half_x - robot_r)).item()
                ry = torch.empty((), device=device).uniform_(cy - (half_y - robot_r), cy + (half_y - robot_r)).item()
                t_center_x = cx + float(table_state[i, 0].item())
                t_center_y = cy + float(table_state[i, 1].item())
                if (rx - t_center_x) ** 2 + (ry - t_center_y) ** 2 >= (robot_r + r_table) ** 2:
                    robot_xy[i, 0], robot_xy[i, 1] = rx, ry
                    placed_robot = True
                    break
            if not placed_robot:
                robot_xy[i, 0] = cx + float(root_state[i, 0].item())
                robot_xy[i, 1] = cy + float(root_state[i, 1].item())

            root_state[i, 0] = robot_xy[i, 0]
            root_state[i, 1] = robot_xy[i, 1]
            root_state[i, 2] = cz + (root_state[i, 2].item() if root_state.ndim == 2 else 0.0)

            # --- Table 固定在默认位姿（叠加 env 平移）---
            table_state[i, 0] = cx + float(table_state[i, 0].item())
            table_state[i, 1] = cy + float(table_state[i, 1].item())
            table_state[i, 2] = cz + float(table_state[i, 2].item())

            # 桌面顶面 z、有效采样矩形（≈ sqrt(0.9) ≈ 0.95 的长宽）
            a_min = table_dims["aabb_min"][i]
            a_max = table_dims["aabb_max"][i]
            width = float((a_max[0] - a_min[0]).item())
            depth = float((a_max[1] - a_min[1]).item())
            half_w_eff = 0.5 * width * 0.95
            half_d_eff = 0.5 * depth * 0.95

            table_center_x = table_state[i, 0].item()
            table_center_y = table_state[i, 1].item()
            table_top_z = table_state[i, 2].item() + 0.5 * float(table_dims["size_aabb"][i, 2].item())

            # 玩具半高 & 放置高度（贴桌面）
            toy_half_h = 0.5 * float(toy_dims["size_aabb"][i, 2].item())
            toy_z = table_top_z + toy_half_h + 0.02

            # --- 目标点（桌面 90% 区域）---
            gx, gy = _sample_xy_box(table_center_x, table_center_y, half_w_eff, half_d_eff)
            goal_xy[i, 0], goal_xy[i, 1] = gx, gy

            # --- 玩具点（也在 90% 区域，且与目标 ≥ 15cm）---
            placed_toy = False
            for _ in range(max_trials):
                tx_s, ty_s = _sample_xy_box(table_center_x, table_center_y, half_w_eff, half_d_eff)
                if (tx_s - gx) ** 2 + (ty_s - gy) ** 2 >= (0.15 ** 2):
                    toy_xy[i, 0], toy_xy[i, 1] = tx_s, ty_s
                    toy_yaw[i] = torch.empty((), device=device).uniform_(-pi, pi)
                    placed_toy = True
                    break
            if not placed_toy:
                toy_xy[i, 0] = table_center_x
                toy_xy[i, 1] = table_center_y
                toy_yaw[i] = 0.0
                use_default_quat_toy[i] = True

            toy_state[i, 0] = toy_xy[i, 0]
            toy_state[i, 1] = toy_xy[i, 1]
            toy_state[i, 2] = toy_z

        # --- 写回仿真 ---
        self.robot.write_root_link_pose_to_sim(root_state[:, :7], env_ids)
        self.robot.write_root_com_velocity_to_sim(root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
        self.table.write_root_pose_to_sim(table_state[:, :7], env_ids)

        # 玩具 yaw-only 四元数（wxyz）
        cos, sin = torch.cos(0.5 * toy_yaw), torch.sin(0.5 * toy_yaw)
        toy_quat_rand = torch.stack([cos, torch.zeros_like(cos), torch.zeros_like(cos), sin], dim=-1)
        toy_quat_default = toy_state[:, 3:7]
        toy_quat = torch.where(use_default_quat_toy.unsqueeze(-1), toy_quat_default, toy_quat_rand)
        toy_root_pose = torch.cat([toy_state[:, :3], toy_quat], dim=-1)
        self.toy.write_root_pose_to_sim(toy_root_pose, env_ids)

        # 目标小球：置于桌面上方 2cm（无碰撞）
        goal_pose = torch.zeros((len(env_ids), 7), device=self.device)
        goal_pose[:, 0] = goal_xy[:, 0]
        goal_pose[:, 1] = goal_xy[:, 1]
        goal_pose[:, 2] = (table_state[:, 2] + 0.5 * table_dims["size_aabb"][:, 2] + 0.02)
        goal_pose[:, 3] = 1.0  # identity quat
        self.goal.write_root_pose_to_sim(goal_pose, env_ids)

        # 缓存/计数清理
        self._goal_xy_w[env_ids] = goal_xy
        self._succ_streak[env_ids] = 0
        if hasattr(self, "_counted_mask"):
            self._counted_mask[env_ids] = False
        if hasattr(self, "_last_obs"):
            for k in self._last_obs:
                self._last_obs[k][env_ids] = 0.0

        # 整体 flag 清零
        self.done_flag = False

    # ----------------- 结束条件（Eval：连续保持） -----------------
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        device = self.device

        # 超时（truncation）
        time_out = (self.episode_length_buf >= self.max_episode_length - 1).to(device)

        # 成功：玩具与目标的 XY 距离
        toy_xy  = self.toy.data.root_pos_w[:, :2]
        goal_xy = self._goal_xy_w
        dist_xy = torch.linalg.norm(toy_xy - goal_xy, dim=-1)
        cond_now = dist_xy < self._thr_xy

        # 连续保持计数
        self._succ_streak = torch.where(cond_now, self._succ_streak + 1, torch.zeros_like(self._succ_streak))
        done_success = self._succ_streak >= self._eval_hold_steps

        # 成功优先，互斥超时
        time_out = time_out & (~done_success)

        # --- 统计（仅新完成那一帧计数）---
        completed_now = done_success | time_out
        if not hasattr(self, "_counted_mask"):
            self._counted_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=device)

        new_mask = completed_now & (~self._counted_mask)
        new_succ = done_success & new_mask
        new_to   = time_out     & new_mask

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

    # ----------------- 单步（Eval：全部完成后统一重置） -----------------
    def step(self, action: torch.Tensor) -> VecEnvStepReturn:
        action = action.to(self.device)
        if self.cfg.action_noise_model:
            action = self._action_noise_model(action)

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

        self.episode_length_buf += 1
        self.common_step_counter += 1

        self.reset_terminated[:], self.reset_time_outs[:] = self._get_dones()
        self.reset_buf = self.reset_terminated | self.reset_time_outs
        self.reward_buf = self._get_rewards()  # 全 0

        # 仅当“所有 env 完成”时，整体重置
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) == self.num_envs:
            self._reset_idx(reset_env_ids)
            self.scene.write_data_to_sim()
            self.sim.forward()
            if self.sim.has_rtx_sensors() and self.cfg.rerender_on_reset:
                self.sim.render()
            self.done_flag = True

        # 事件
        if self.cfg.events and "interval" in self.event_manager.available_modes:
            self.event_manager.apply(mode="interval", dt=self.step_dt)

        # 观测
        self.obs_buf = self._get_observations()
        if self.cfg.observation_noise_model:
            self.obs_buf["policy"] = self._observation_noise_model(self.obs_buf["policy"])

        return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras

    # ----------------- 动作（完成 env 冻结当前姿态） -----------------
    def _pre_physics_step(self, actions: torch.Tensor):
        self.actions = actions
        self.actions = torch.nan_to_num(self.actions, nan=0.0, posinf=0.0, neginf=0.0)
        self.actions.clamp_(-1.0, 1.0)

        # 冻结已完成 env 的动作，避免扰动统计
        completed = getattr(self, "_counted_mask", None)
        if completed is not None and completed.any():
            q_current = self.robot.data.joint_pos
            a_freeze = (q_current - self.action_offset) / (self.action_scale + 1e-8)
            a_freeze = a_freeze.clamp_(-1.0, 1.0)
            self.actions[completed] = a_freeze[completed]

    def _apply_action(self):
        target = self.action_offset + self.action_scale * self.actions
        self.robot.set_joint_position_target(target)

    # ----------------- 观测（与训练一致） -----------------
    def _get_observations(self) -> VecEnvObs:
        device = self.device
        q  = self.robot.data.joint_pos
        qd = self.robot.data.joint_vel

        # 参考刚体：优先 cfg.reference_body
        ref_name = getattr(self.cfg, "reference_body", None)
        names = getattr(self.robot.data, "body_names", self.robot.data.body_names)
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

        ex = torch.zeros_like(root_pos_w); ex[:, 0] = 1.0
        ez = torch.zeros_like(root_pos_w); ez[:, 2] = 1.0
        tangent = quat_apply(root_quat_w, ex)
        normal  = quat_apply(root_quat_w, ez)
        root_rot_6d = torch.cat([tangent, normal], dim=-1)
        root_z = root_pos_w[:, 2:3]

        toy_pos_w  = self.toy.data.root_pos_w
        toy_quat_w = self.toy.data.root_quat_w
        toy_center_rel = toy_pos_w - root_pos_w

        # toy OBB 尺寸缓存
        if not hasattr(self, "_cached_toy_size_obb") or self._cached_toy_size_obb.shape[0] != self.num_envs:
            self._cached_toy_size_obb = self._get_dims_for(self.toy, None)["size_obb"].to(device)
        toy_size_obb = self._cached_toy_size_obb

        # goal 相对 toy 的 XY 向量
        goal_vec_xy = self._goal_xy_w - toy_pos_w[:, :2]

        self_part  = torch.cat([q, qd, root_z, root_rot_6d, root_lin_w, root_ang_w], dim=-1)
        inter_part = torch.cat([toy_center_rel, toy_size_obb, toy_quat_w], dim=-1)
        goal_part  = goal_vec_xy  # (N,2)
        policy_obs = torch.cat([self_part, inter_part, goal_part], dim=-1)

        obs = {"policy": torch.nan_to_num(policy_obs)}

        # 完成 env 使用上一帧观测，避免抖动
        if not hasattr(self, "_last_obs"):
            self._last_obs = {k: v.clone() for k, v in obs.items()}
        completed = getattr(self, "_counted_mask", None)
        if completed is not None and completed.any():
            obs["policy"][completed] = self._last_obs["policy"][completed]
        self._last_obs = {k: v.clone() for k, v in obs.items()}
        return obs

    # ----------------- 评测用 flag 接口 -----------------
    def get_done_flag(self):
        return self.done_flag

    def set_done_flag(self, new_flag: bool):
        self.done_flag = bool(new_flag)

    # ----------------- 评测奖励：恒为 0 -----------------
    def _get_rewards(self) -> torch.Tensor:
        return torch.zeros(self.num_envs, device=self.device)
