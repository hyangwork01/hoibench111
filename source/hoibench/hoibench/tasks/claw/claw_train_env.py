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

    cfg: ClawEnvCfg

    def __init__(self, cfg: ClawEnvCfg, render_mode: str | None = None, **kwargs):
        self.env_spacing = cfg.scene.env_spacing
        self.env_sample_len = self.env_spacing - 2

        super().__init__(cfg, render_mode, **kwargs)

                                         
        ndof = self.robot.data.joint_pos.shape[1]
        base_self_dim = 2 * ndof + 1 + 6 + 3 + 3   # q, qd, root_z, root_rot6d, root_lin, root_ang
        inter_dim = 10                              # toy_center_rel(3)+size_obb(3)+toy_quat(4)
        goal_dim = 2                                                                  
        obs_dim = base_self_dim + inter_dim + goal_dim

        self.cfg.action_space = ndof
        self.cfg.observation_space = obs_dim
        self._configure_gym_env_spaces()

              
        dof_lower = self.robot.data.soft_joint_pos_limits[0, :, 0]
        dof_upper = self.robot.data.soft_joint_pos_limits[0, :, 1]
        self.action_offset = 0.5 * (dof_upper + dof_lower)
        self.action_scale = (dof_upper - dof_lower).clamp_min(1e-6)

            

                           
        self._prev_xy_dist = torch.full((self.num_envs,), -1.0, device=self.device)

                     
        self._goal_xy_w = torch.zeros((self.num_envs, 2), device=self.device)
        self.goal_success_xy = 0.03

                                              
    def _setup_scene(self):
                
        self.robot = Articulation(self.cfg.robot)
        self.table = RigidObject(self.cfg.table)
        self.toy   = RigidObject(self.cfg.toy)
        self.goal  = RigidObject(self.cfg.goal)                                

                   
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
        self.scene.rigid_objects["table"] = self.table
        self.scene.rigid_objects["toy"]   = self.toy
        self.scene.rigid_objects["goal"]  = self.goal

            
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

                                                                              
    def _get_dims_for(self, obj: RigidObject, env_ids: Sequence[int] | None = None):
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

            
        table_dims = self._get_dims_for(self.table, env_ids)
        toy_dims   = self._get_dims_for(self.toy,   env_ids)

              
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

                   
        def _sample_xy_box(cx: float, cy: float, half_w: float, half_d: float):
            rx = torch.empty((), device=device).uniform_(cx - half_w, cx + half_w).item()
            ry = torch.empty((), device=device).uniform_(cy - half_d, cy + half_d).item()
            return rx, ry

        for i in range(N):
            cx, cy, cz = origins[i].tolist()

                                     
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

                                               
            table_state[i, 0] = cx + float(table_state[i, 0].item())
            table_state[i, 1] = cy + float(table_state[i, 1].item())
            table_state[i, 2] = cz + float(table_state[i, 2].item())

                                                   
            a_min = table_dims["aabb_min"][i]
            a_max = table_dims["aabb_max"][i]
            width = float((a_max[0] - a_min[0]).item())
            depth = float((a_max[1] - a_min[1]).item())
            half_w_eff = 0.5 * width * 0.95
            half_d_eff = 0.5 * depth * 0.95

            table_center_x = table_state[i, 0].item()
            table_center_y = table_state[i, 1].item()
            table_top_z = table_state[i, 2].item() + 0.5 * float(table_dims["size_aabb"][i, 2].item())

                               
            toy_half_h = 0.5 * float(toy_dims["size_aabb"][i, 2].item())
            toy_z = table_top_z + toy_half_h + 0.02

                                     
            gx, gy = _sample_xy_box(table_center_x, table_center_y, half_w_eff, half_d_eff)
            goal_xy[i, 0], goal_xy[i, 1] = gx, gy

                                        
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

                      
        self.robot.write_root_link_pose_to_sim(root_state[:, :7], env_ids)
        self.robot.write_root_com_velocity_to_sim(root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        self.table.write_root_pose_to_sim(table_state[:, :7], env_ids)

                               
        cos, sin = torch.cos(0.5 * toy_yaw), torch.sin(0.5 * toy_yaw)
        toy_quat_rand = torch.stack([cos, torch.zeros_like(cos), torch.zeros_like(cos), sin], dim=-1)
        toy_quat_default = toy_state[:, 3:7]
        toy_quat = torch.where(use_default_quat_toy.unsqueeze(-1), toy_quat_default, toy_quat_rand)
        toy_root_pose = torch.cat([toy_state[:, :3], toy_quat], dim=-1)
        self.toy.write_root_pose_to_sim(toy_root_pose, env_ids)

                                
        goal_pose = torch.zeros((len(env_ids), 7), device=self.device)
        goal_pose[:, 0] = goal_xy[:, 0]
        goal_pose[:, 1] = goal_xy[:, 1]
        goal_pose[:, 2] = (table_state[:, 2] + 0.5 * table_dims["size_aabb"][:, 2] + 0.02)
        goal_pose[:, 3] = 1.0  # identity quat
        self.goal.write_root_pose_to_sim(goal_pose, env_ids)

                 
        self._goal_xy_w[env_ids] = goal_xy
        self._prev_xy_dist[env_ids] = -1.0

                                
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        device = self.device
            
        time_out = (self.episode_length_buf >= self.max_episode_length - 1).to(device)

        toy_xy  = self.toy.data.root_pos_w[:, :2]
        goal_xy = self._goal_xy_w
        dist_xy = torch.linalg.norm(toy_xy - goal_xy, dim=-1)

        thr = self.goal_success_xy         

        done_success = (dist_xy < thr)
        time_out = time_out & (~done_success)

                        
        return done_success, time_out


                
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
        self.reward_buf = self._get_rewards()

                  
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self._reset_idx(reset_env_ids)
            self.scene.write_data_to_sim()
            self.sim.forward()
            if self.sim.has_rtx_sensors() and self.cfg.rerender_on_reset:
                self.sim.render()

            
        self.obs_buf = self._get_observations()
        if self.cfg.observation_noise_model:
            self.obs_buf["policy"] = self._observation_noise_model(self.obs_buf["policy"])

        return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras

                
    def _apply_action(self):
        target = self.action_offset + self.action_scale * self.actions
        self.robot.set_joint_position_target(target)

                                                         
    def _get_observations(self) -> VecEnvObs:
        device = self.device
        ndof = self.robot.data.joint_pos.shape[1]
        q  = self.robot.data.joint_pos
        qd = self.robot.data.joint_vel

                                    
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

                         
        if not hasattr(self, "_cached_toy_size_obb") or self._cached_toy_size_obb.shape[0] != self.num_envs:
            self._cached_toy_size_obb = self._get_dims_for(self.toy, None)["size_obb"].to(device)
        toy_size_obb = self._cached_toy_size_obb

                                        
        goal_vec_xy = self._goal_xy_w - toy_pos_w[:, :2]

        self_part  = torch.cat([q, qd, root_z, root_rot_6d, root_lin_w, root_ang_w], dim=-1)
        inter_part = torch.cat([toy_center_rel, toy_size_obb, toy_quat_w], dim=-1)
        goal_part  = goal_vec_xy  # (N,2)

        policy_obs = torch.cat([self_part, inter_part, goal_part], dim=-1)

        if isinstance(self.cfg.observation_space, int):
            assert policy_obs.shape[1] == self.cfg.observation_space,\
                f"Obs length mismatch: got {policy_obs.shape[1]}, expect {self.cfg.observation_space}"

        obs = {"policy": torch.nan_to_num(policy_obs)}
        return obs

                  
    def _pre_physics_step(self, actions: torch.Tensor):
        self.actions = actions
        self.actions = torch.nan_to_num(self.actions, nan=0.0, posinf=0.0, neginf=0.0)
        self.actions.clamp_(-1.0, 1.0)

                                        
    def _get_rewards(self) -> torch.Tensor:
        device = self.device
        eps = 1e-6

                     
        toy_xy  = self.toy.data.root_pos_w[:, :2]
        goal_xy = self._goal_xy_w
        dist_goal = torch.linalg.norm(toy_xy - goal_xy, dim=-1)

                         
        sigma_goal = float(getattr(self.cfg, "goal_reward_sigma", 0.30))
        w_goal     = float(getattr(self.cfg, "w_goal", 2.0))
        r_goal_xy  = w_goal * torch.exp(-dist_goal / (sigma_goal + eps))

              
        w_action = float(getattr(self.cfg, "w_action", 2.5e-3))
        w_qd     = float(getattr(self.cfg, "w_qd", 1.0e-4))
        w_limits = float(getattr(self.cfg, "w_limits", 2.0e-2))
        step_pen = float(getattr(self.cfg, "step_time_pen", 0.01))

        p_action = w_action * (self.actions ** 2).sum(dim=-1)
        qd       = self.robot.data.joint_vel
        p_qd     = w_qd * (qd ** 2).sum(dim=-1)
        lower    = self.robot.data.soft_joint_pos_limits[0, :, 0]
        upper    = self.robot.data.soft_joint_pos_limits[0, :, 1]
        q        = self.robot.data.joint_pos
        rel      = (q - lower) / (upper - lower + eps)
        margin   = torch.clamp(0.95 - torch.minimum(rel, 1 - rel), min=0.0)
        p_limits = w_limits * (margin ** 2).sum(dim=-1)

        reward = r_goal_xy - p_action - p_qd - p_limits - step_pen
        reward = torch.nan_to_num(reward, nan=0.0, posinf=0.0, neginf=0.0)

            
        self.extras["r_terms"] = {
            "goal_xy":  r_goal_xy.mean().item(),
            "p_action": p_action.mean().item(),
            "p_qd":     p_qd.mean().item(),
            "p_limits": p_limits.mean().item(),
            "dist_xy":  dist_goal.mean().item(),
        }
        return reward
