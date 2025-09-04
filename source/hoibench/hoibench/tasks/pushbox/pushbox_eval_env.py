from __future__ import annotations
import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import quat_apply
from isaaclab.envs.common import VecEnvObs, VecEnvStepReturn
from collections.abc import Sequence
from math import pi

from ..sitchair.base import HOIEnv
from .pushbox_cfg import PushboxEnvCfg


class PushboxEnv(HOIEnv):
    cfg: PushboxEnvCfg

    def __init__(self, cfg: PushboxEnvCfg, render_mode: str | None = None, **kwargs):
        self.env_spacing = cfg.scene.env_spacing
        self.env_sample_len = self.env_spacing - 2

        super().__init__(cfg, render_mode, **kwargs)

        # dimensions
        ndof = self.robot.data.joint_pos.shape[1]
        base_self_dim = 2 * ndof + 1 + 6 + 3 + 3   # q, qd, root_z, root_rot6d, root_lin, root_ang
        inter_dim = 10                              # obj_center_rel(3)+size_obb(3)+obj_quat(4)
        obs_dim = base_self_dim + inter_dim + 2

        self.cfg.action_space = ndof
        self.cfg.observation_space = obs_dim
        self._configure_gym_env_spaces()

        # action scale/offset
        dof_lower = self.robot.data.soft_joint_pos_limits[0, :, 0]
        dof_upper = self.robot.data.soft_joint_pos_limits[0, :, 1]
        self.action_offset = 0.5 * (dof_upper + dof_lower)
        self.action_scale = (dof_upper - dof_lower).clamp_min(1e-6)

        # bookkeeping
        self._obj_init_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self._target_xy = torch.zeros((self.num_envs, 2), device=self.device)

        self._counted_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.stat_success_count = 0
        self.stat_timeout_count = 0
        self.stat_success_time_sum = 0.0
        self.stat_timeout_time_sum = 0.0
        self.stat_completed = 0
        self.stat_avg_time = 0.0

    # ----------------- scene -----------------
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

    # ----------------- reset (eval: whole batch when all done) -----------------
    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, dtype=torch.int32, device=self.device)
        else:
            env_ids = torch.as_tensor(env_ids, dtype=torch.int32, device=self.device).contiguous()

        self.episode_length_buf[env_ids] = 0

        device = self.device
        half_x = half_y = 0.5 * float(self.env_sample_len)
        robot_r = 1.0
        max_trials = 1000
        yaw_range = pi

        obj_info = self._get_dims(env_ids)
        obj_r_all = obj_info["keepout_radius_xy"]  # (N,)

        root_state = self.robot.data.default_root_state[env_ids].clone()
        joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        joint_vel = self.robot.data.default_joint_vel[env_ids].clone()
        obj_state = self.obj.data.default_root_state[env_ids].clone()

        ground_clearance = 0.05
        obj_state[:, 2] =obj_state[:, 2] - obj_info["aabb_min"][:, 2] + ground_clearance


        origins = self.scene.env_origins[env_ids]

        N = len(env_ids)
        robot_xy = torch.zeros((N, 2), device=device)
        obj_xy = torch.zeros((N, 2), device=device)
        obj_yaw = torch.zeros((N,), device=device)
        use_default_quat = torch.zeros((N,), dtype=torch.bool, device=device)

        def _sample_xy(cx: float, cy: float, r_keepout: float):
            x_lo, x_hi = cx - (half_x - r_keepout), cx + (half_x - r_keepout)
            y_lo, y_hi = cy - (half_y - r_keepout), cy + (half_y - r_keepout)
            if x_hi < x_lo or y_hi < y_lo:
                return cx, cy
            rx = torch.empty((), device=device).uniform_(x_lo, x_hi).item()
            ry = torch.empty((), device=device).uniform_(y_lo, y_hi).item()
            return rx, ry

        for i in range(N):
            cx, cy, cz = origins[i].tolist()
            r_obj = float(obj_r_all[i].item())

            placed = False
            for _ in range(max_trials):
                rx, ry = _sample_xy(cx, cy, robot_r)
                ox, oy = _sample_xy(cx, cy, r_obj)
                if ((rx - ox) ** 2 + (ry - oy) ** 2) >= (robot_r + r_obj) ** 2:
                    robot_xy[i, 0], robot_xy[i, 1] = rx, ry
                    obj_xy[i, 0], obj_xy[i, 1] = ox, oy
                    obj_yaw[i] = torch.empty((), device=device).uniform_(-yaw_range, yaw_range)
                    placed = True
                    break

            if not placed:
                robot_xy[i, 0] = cx + float(root_state[i, 0].item())
                robot_xy[i, 1] = cy + float(root_state[i, 1].item())
                obj_xy[i, 0] = cx + float(obj_state[i, 0].item())
                obj_xy[i, 1] = cy + float(obj_state[i, 1].item())
                obj_yaw[i] = 0.0
                use_default_quat[i] = True

            world_z = cz + root_state[i, 2].item()
            root_state[i, 0] = robot_xy[i, 0]
            root_state[i, 1] = robot_xy[i, 1]
            root_state[i, 2] = world_z

            obj_world_z = cz + obj_state[i, 2].item()
            obj_state[i, 0] = obj_xy[i, 0]
            obj_state[i, 1] = obj_xy[i, 1]
            obj_state[i, 2] = obj_world_z

        # write sim states
        self.robot.write_root_link_pose_to_sim(root_state[:, :7], env_ids)
        self.robot.write_root_com_velocity_to_sim(root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        cos, sin = torch.cos(0.5 * obj_yaw), torch.sin(0.5 * obj_yaw)
        obj_quat_rand = torch.stack([cos, torch.zeros_like(cos), torch.zeros_like(cos), sin], dim=-1)
        obj_quat_default = obj_state[:, 3:7]
        obj_quat = torch.where(use_default_quat.unsqueeze(-1), obj_quat_default, obj_quat_rand)

        obj_root_pose = torch.cat([obj_state[:, :3], obj_quat], dim=-1)
        self.obj.write_root_pose_to_sim(obj_root_pose, env_ids)

        push_off = self.cfg.push_target_offset
        fwd_xy = torch.stack([torch.cos(obj_yaw), torch.sin(obj_yaw)], dim=-1)
        self._target_xy[env_ids] = obj_state[:, :2] + push_off * fwd_xy

        # cache initial box position for success check
        self._obj_init_pos[env_ids] = obj_state[:, :3]

        # clear masks and caches

        self._counted_mask[env_ids] = False

    # helper for box dims/keepout
    def _get_dims(self, env_ids: Sequence[int] | None = None):
        import numpy as np
        import isaacsim.core.utils.bounds as bounds_utils
        import isaacsim.core.utils.stage as stage_utils

        device = self.device
        if env_ids is None:
            env_ids = list(range(self.num_envs))
        else:
            env_ids = list(env_ids) or [0]

        pat = self.obj.cfg.prim_path
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
                suffix = pat.lstrip("/")

        env_ns = getattr(self.scene, "env_ns", "/World/envs")
        prim_paths = [f"{env_ns}/env_{int(i)}/{suffix}" for i in env_ids]

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

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        device = self.device

        # time out
        time_out = (self.episode_length_buf >= self.max_episode_length - 1).to(device)

        # success: box displacement from its initial position exceeds threshold
        disp = torch.linalg.norm(self.obj.data.root_pos_w - self._obj_init_pos, dim=-1)
        thr = float(getattr(self.cfg, "push_target_offset", 0.5))
        done_success = disp >= thr

        time_out = time_out & (~done_success)

        completed_now = done_success | time_out
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

    def step(self, action: torch.Tensor) -> VecEnvStepReturn:
        action = action.to(self.device)
        if self.cfg.action_noise_model:
            action = self._action_noise_model(action)

        # pre-physics: clamp/sanitize
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
        self.reward_buf = self._get_rewards()

        # eval: reset all when everyone is done
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) == self.num_envs:
            self._reset_idx(reset_env_ids)
            self.scene.write_data_to_sim()
            self.sim.forward()
            if self.sim.has_rtx_sensors() and self.cfg.rerender_on_reset:
                self.sim.render()

        if self.cfg.events and "interval" in self.event_manager.available_modes:
            self.event_manager.apply(mode="interval", dt=self.step_dt)

        self.obs_buf = self._get_observations()
        if self.cfg.observation_noise_model:
            self.obs_buf["policy"] = self._observation_noise_model(self.obs_buf["policy"])

        return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras

    def _pre_physics_step(self, actions: torch.Tensor):
        # sanitize and clamp
        self.actions = torch.nan_to_num(actions, nan=0.0, posinf=0.0, neginf=0.0)
        self.actions.clamp_(-1.0, 1.0)

        # freeze completed envs (hold current joint pose)
        completed = getattr(self, "_counted_mask", None)
        if completed is not None and completed.any():
            q_current = self.robot.data.joint_pos
            a_freeze = (q_current - self.action_offset) / (self.action_scale + 1e-8)
            a_freeze = a_freeze.clamp_(-1.0, 1.0)
            self.actions[completed] = a_freeze[completed]

    def _get_observations(self) -> VecEnvObs:
        device = self.device
        q = self.robot.data.joint_pos
        qd = self.robot.data.joint_vel

        # choose reference body for root quantities
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
        root_rot_6d = torch.cat([tangent, normal], dim=-1)
        root_z = root_pos_w[:, 2:3]

        obj_pos_w = self.obj.data.root_pos_w
        obj_quat_w = self.obj.data.root_quat_w
        obj_center_rel = obj_pos_w - root_pos_w

        if not hasattr(self, "_cached_obj_size_obb") or self._cached_obj_size_obb.shape[0] != self.num_envs:
            self._cached_obj_size_obb = self._get_dims(None)["size_obb"].to(device)
        obj_size_obb = self._cached_obj_size_obb

        self_part = torch.cat([q, qd, root_z, root_rot_6d, root_lin_w, root_ang_w], dim=-1)
        inter_part = torch.cat([obj_center_rel, obj_size_obb, obj_quat_w], dim=-1)
        goal_xy = (self._target_xy - obj_pos_w[:, :2])
        policy_obs = torch.cat([self_part, inter_part, goal_xy], dim=-1)

        obs = {"policy": torch.nan_to_num(policy_obs)}
        if not hasattr(self, "_last_obs"):
            self._last_obs = {k: v.clone() for k, v in obs.items()}
        completed = getattr(self, "_counted_mask", None)
        if completed is not None and completed.any():
            obs["policy"][completed] = self._last_obs["policy"][completed]
        return obs

    def _apply_action(self):
        target = self.action_offset + self.action_scale * self.actions
        self.robot.set_joint_position_target(target)



    def _get_rewards(self) -> torch.Tensor:
        # eval: zero rewards
        return torch.zeros(self.num_envs, device=self.device)
