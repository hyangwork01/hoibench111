# Copyright (c) 2022-2025
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations
import torch

from isaaclab.envs import DirectRLEnv
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.envs.common import VecEnvObs, VecEnvStepReturn


class HOIEnvCfg(DirectRLEnvCfg):
    seed: int = 0
    # --- 每秒的帧率:dt*decimation ---
    decimation: int = 2
    dt: float = 1 / 120
    episode_length_s: float = 20.0

    # --- 场景 ---
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=12.0, replicate_physics=True)

    # --- 物理/仿真 ---
    sim: SimulationCfg = SimulationCfg(
        dt=dt,
        render_interval=decimation,
        physx=PhysxCfg(
            gpu_found_lost_pairs_capacity=2**23,
            gpu_total_aggregate_pairs_capacity=2**23,
        ),
    )

    # 先占位
    observation_space = 1
    action_space = 1
    state_space = 0


class HOIEnv(DirectRLEnv):
    cfg: HOIEnvCfg
    def __init__(self, cfg: HOIEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

    def _setup_scene(self):
        super()._setup_scene()


    def _apply_action(self):
        super._apply_action()

    def _get_observations(self) -> VecEnvObs:
        super._get_observations()

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        super._get_dones()

    def _reset_idx(self, env_ids: torch.Tensor | None):
        super._reset_idx(env_ids)

