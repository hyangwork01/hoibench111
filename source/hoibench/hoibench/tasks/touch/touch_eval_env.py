from __future__ import annotations
import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import quat_apply
from isaaclab.envs.common import VecEnvObs, VecEnvStepReturn
from collections.abc import Sequence
from math import pi

from .base import HOIEnv
from .touch_cfg import TouchEnvCfg


class TouchEnv(HOIEnv):
    """Eval 版 touch：与训练版保持同一观测与成功判据；仅在所有 env 完成时整体 reset。"""
    cfg: TouchEnvCfg

    def __init__(self, cfg: TouchEnvCfg, render_mode: str | None = None, **kwargs):
        self.env_spacing = cfg.scene.env_spacing
        self.env_sample_len = self.env_spacing - 2
        super().__init__(cfg, render_mode, **kwargs)

        # === 观测维度：自身体态 + goal_rel(3) ===
        ndof = self.robot.data.joint_pos.shape[1]
        base_self_dim = 2 * ndof + 1 + 6 + 3 + 3   # q, qd, root_z, root_rot6d, root_lin, root_ang
        goal_dim = 3
        obs_dim = base_self_dim + goal_dim

        self.cfg.action_space = ndof
        self.cfg.observation_space = obs_dim
        self._configure_gym_env_spaces()  # Gym spaces 配置；见 DirectRLEnv 文档。:contentReference[oaicite:1]{index=1}

        # 动作缩放
        dof_lower = self.robot.data.soft_joint_pos_limits[0, :, 0]
        dof_upper = self.robot.data.soft_joint_pos_limits[0, :, 1]
        self.action_offset = 0.5 * (dof_upper + dof_lower)
        self.action_scale = (dof_upper - dof_lower).clamp_min(1e-6)

        # 统计与缓存
        self.done_flag = False
        self._counted_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._last_obs = None

        # 目标点缓存
        self.goal_pos_w = torch.zeros((self.num_envs, 3), device=self.device)

    # ----------------- 场景构建 -----------------
    def _setup_scene(self):
        # 机器人
        self.robot = Articulation(self.cfg.robot)
        # 目标小球（仅可见、不可碰撞、运动学体；在 cfg.goal 里定义）
        self.goal = RigidObject(self.cfg.goal)  # SphereCfg + PreviewSurfaceCfg 上色为红色。:contentReference[oaicite:2]{index=2}

        # 地面
        spawn_ground_plane(
            prim_path="/World/ground",
            cfg=GroundPlaneCfg(
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    static_friction=1.0, dynamic_friction=1.0, restitution=0.0
                ),
            ),
        )

        # 克隆 envs + 过滤地面全局碰撞
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=["/World/ground"])

        # 注册
        self.scene.articulations["robot"] = self.robot
        self.scene.rigid_objects["goal"] = self.goal

        # 光照
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    # ----------------- 重置（Eval：单次采样 robot 位姿 + 目标点） -----------------
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
        yaw_range = pi
        max_trials = 1000

        # 默认状态（局部坐标）
        root_state = self.robot.data.default_root_state[env_ids].clone()  # (N,13)
        joint_pos  = self.robot.data.default_joint_pos[env_ids].clone()
        joint_vel  = self.robot.data.default_joint_vel[env_ids].clone()
        origins = self.scene.env_origins[env_ids]  # (N,3)

        N = len(env_ids)
        robot_xy = torch.zeros((N, 2), device=device)

        def _sample_xy(cx: float, cy: float, r_keepout: float = 1.0):
            x_lo, x_hi = cx - (half_x - r_keepout), cx + (half_x - r_keepout)
            y_lo, y_hi = cy - (half_y - r_keepout), cy + (half_y - r_keepout)
            rx = torch.empty((), device=device).uniform_(x_lo, x_hi).item()
            ry = torch.empty((), device=device).uniform_(y_lo, y_hi).item()
            return rx, ry

        # 放置机器人根位置 + yaw
        for i in range(N):
            cx, cy, cz = origins[i].tolist()
            rx, ry = _sample_xy(cx, cy)
            robot_xy[i, 0], robot_xy[i, 1] = rx, ry
            world_z = cz + (root_state[i, 2].item() if root_state.ndim == 2 else 0.0)
            root_state[i, 0] = rx
            root_state[i, 1] = ry
            root_state[i, 2] = world_z

        yaw = torch.empty((N,), device=device).uniform_(-yaw_range, yaw_range)
        cos, sin = torch.cos(0.5 * yaw), torch.sin(0.5 * yaw)
        quat = torch.stack([cos, torch.zeros_like(cos), torch.zeros_like(cos), sin], dim=-1)  # wxyz

        self.robot.write_root_link_pose_to_sim(torch.cat([root_state[:, :3], quat], dim=-1), env_ids)
        self.robot.write_root_com_velocity_to_sim(root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        # === 目标点采样（围绕根/参考体，半径 ~ approx_reach） ===
        approx_R = float(getattr(self.cfg, "approx_reach", 1.0))
        approx_R = max(0.2, approx_R)  # 最少给 20cm

        goal_pose = torch.zeros((N, 7), device=device)
        goal_pose[:, 3] = 1.0  # 单位四元数（wxyz）

        for i in range(N):
            cx, cy, cz = origins[i].tolist()
            rx, ry, rz = root_state[i, 0].item(), root_state[i, 1].item(), root_state[i, 2].item()
            # z：地面 5cm ~ root 上方 approx_R
            z_lo = cz + 0.05
            z_hi = rz + approx_R
            if z_hi <= z_lo:
                z_hi = z_lo + 0.10
            gz = torch.empty((), device=device).uniform_(z_lo, z_hi).item()
            # 平面极坐标
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

        # 清理标志
        self._counted_mask[env_ids] = False
        self.done_flag = False

    # ----------------- 成功/超时（与训练一致的“触达点”标准） -----------------
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        device = self.device

        # 超时
        time_out = (self.episode_length_buf >= self.max_episode_length - 1).to(device)

        # 成功：接触体到目标点的 3D 距离 < 阈值，且超过冷启动步数
        thr = float(getattr(self.cfg, "touch_threshold", 0.03))     # 3cm
        min_success_steps = int(getattr(self.cfg, "min_success_steps", 5))
        allow_success = (self.episode_length_buf >= min_success_steps)

        names = self.robot.data.body_names
        contact_idx = names.index(self.cfg.contact_body)
        contact_pos = self.robot.data.body_link_pos_w[:, contact_idx]  # (N,3)

        dist = torch.linalg.norm(contact_pos - self.goal_pos_w, dim=-1)  # (N,)
        done_success = (dist < thr) & allow_success

        # 互斥
        time_out = time_out & (~done_success)

        # 只对新完成 env 计数一次
        completed_now = done_success | time_out
        new_mask = completed_now & (~self._counted_mask)
        self._counted_mask |= completed_now  # 冻结这些 env

        return done_success, time_out

    # ----------------- 单步（Eval：所有 env 完成后整体 reset） -----------------
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
        self.reward_buf = self._get_rewards()  # eval: 全 0

        # 仅当“所有 env 完成”时整体 reset
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) == self.num_envs:
            self._reset_idx(reset_env_ids)
            self.scene.write_data_to_sim()
            self.sim.forward()
            if self.sim.has_rtx_sensors() and self.cfg.rerender_on_reset:
                self.sim.render()
            self.done_flag = True

        # 事件/周期性逻辑
        if self.cfg.events:
            if "interval" in self.event_manager.available_modes:
                self.event_manager.apply(mode="interval", dt=self.step_dt)

        # 观测
        self.obs_buf = self._get_observations()
        if self.cfg.observation_noise_model:
            self.obs_buf["policy"] = self._observation_noise_model(self.obs_buf["policy"])

        return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras

    # ----------------- 预物理步：健壮化 + 冻结完成 env 的动作 -----------------
    def _pre_physics_step(self, actions: torch.Tensor):
        # 正确接管外部 action
        self.actions = actions
        # 数值健壮化与裁剪
        self.actions = torch.nan_to_num(self.actions, nan=0.0, posinf=0.0, neginf=0.0)
        self.actions.clamp_(-1.0, 1.0)

        # 已完成 env：将目标动作设为“维持当前关节”
        completed = getattr(self, "_counted_mask", None)
        if completed is not None and completed.any():
            q_current = self.robot.data.joint_pos
            a_hold = (q_current - self.action_offset) / (self.action_scale + 1e-8)
            a_hold = a_hold.clamp_(-1.0, 1.0)
            self.actions[completed] = a_hold[completed]

    # ----------------- 观测：与训练一致 -----------------
    def _get_observations(self) -> VecEnvObs:
        ndof = self.robot.data.joint_pos.shape[1]
        q  = self.robot.data.joint_pos
        qd = self.robot.data.joint_vel

        # 参考刚体：优先 cfg.reference_body -> root_link
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

        # 连续 6D 姿态 + root_z
        ex = torch.zeros_like(root_pos_w); ex[:, 0] = 1.0
        ez = torch.zeros_like(root_pos_w); ez[:, 2] = 1.0
        tangent = quat_apply(root_quat_w, ex)
        normal  = quat_apply(root_quat_w, ez)
        root_rot_6d = torch.cat([tangent, normal], dim=-1)
        root_z = root_pos_w[:, 2:3]

        # 目标相对位移（相对参考刚体）
        goal_rel = self.goal_pos_w - root_pos_w  # (N,3)

        self_part  = torch.cat([q, qd, root_z, root_rot_6d, root_lin_w, root_ang_w], dim=-1)
        policy_obs = torch.cat([self_part, goal_rel], dim=-1)

        obs = {"policy": torch.nan_to_num(policy_obs)}

        # 完成 env 复用上一帧观测以稳定可视化
        if self._last_obs is not None:
            completed = getattr(self, "_counted_mask", None)
            if completed is not None and completed.any():
                obs["policy"][completed] = self._last_obs["policy"][completed]
        self._last_obs = {k: v.clone() for k, v in obs.items()}
        return obs

    def _apply_action(self):
        target = self.action_offset + self.action_scale * self.actions
        self.robot.set_joint_position_target(target)

    # Eval 便利接口
    def get_done_flag(self):
        return self.done_flag

    def set_done_flag(self, new_flag):
        self.done_flag = new_flag

    # Eval：奖励不参与决策
    def _get_rewards(self) -> torch.Tensor:
        return torch.zeros(self.num_envs, device=self.device)
