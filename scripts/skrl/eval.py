# Copyright (c) 2022-2025
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

import argparse
import os
import sys
import time
import torch
import gymnasium as gym
from packaging import version

from isaaclab.app import AppLauncher


# ---- CLI ----
parser = argparse.ArgumentParser(description="Evaluate an RL checkpoint on IsaacLab env.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)

# === 改动 1：将 checkpoint / checkpoint_yaml 设为必填 ===
parser.add_argument(
    "--checkpoint",
    type=str,
    required=True,
    help="Path to the skrl agent checkpoint file saved during training (e.g., agent.pt / agent.pth).",
)
parser.add_argument(
    "--checkpoint_yaml",
    type=str,
    required=True,
    help="Path to the skrl Runner/agent YAML config saved during training (e.g., agent.yaml).",
)

parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument(
    "--ml_framework",
    type=str,
    default="torch",
    choices=["torch", "jax", "jax-numpy"],
    help="The ML framework used for training the skrl agent.",
)
parser.add_argument(
    "--algorithm",
    type=str,
    default="PPO",
    choices=["AMP", "PPO", "IPPO", "MAPPO"],
    help="The RL algorithm used for training the skrl agent.",
)

parser.add_argument(
    "--rounds",
    type=int,
    default=5,
    help="Number of episodes per environment (total = rounds * num_envs)."
)


# 交给 AppLauncher 的参数
AppLauncher.add_app_launcher_args(parser)

# 解析
args_cli, hydra_args = parser.parse_known_args()

# 清理 argv 给 Hydra
sys.argv = [sys.argv[0]] + hydra_args

# 启动 Omniverse
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab_tasks.utils.hydra import hydra_task_config
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_rl.skrl import SkrlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
import hoibench.tasks  # noqa: F401
import skrl
# 版本检查
SKRL_VERSION = "1.4.2"
if version.parse(skrl.__version__) < version.parse(SKRL_VERSION):
    raise RuntimeError(
        f"Unsupported skrl version: {skrl.__version__}. "
        f"Install supported version using 'pip install skrl>={SKRL_VERSION}'"
    )

if args_cli.ml_framework.startswith("torch"):
    from skrl.utils.runner.torch import Runner
elif args_cli.ml_framework.startswith("jax"):
    from skrl.utils.runner.jax import Runner


def _unwrap_env(env):
    """取到底层 DirectRLEnv / 你的自定义 env 实例。"""
    base = getattr(env, "unwrapped", env)
    # 尽力剥一剥 wrapper（Gym / IsaacLab / Skrl）
    seen = set()
    while hasattr(base, "unwrapped") and id(base.unwrapped) not in seen:
        seen.add(id(base))
        if base.unwrapped is base:
            break
        base = base.unwrapped
    return base


@hydra_task_config(args_cli.task, agent_cfg_entry_point=None)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, experiment_cfg: dict):
    # 覆盖 cfg
    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device

    # skrl 后端
    if args_cli.ml_framework.startswith("jax"):
        skrl.config.jax.backend = "jax" if args_cli.ml_framework == "jax" else "numpy"

    algorithm = args_cli.algorithm.lower()

    # === 改动 2：直接使用必填参数 checkpoint / checkpoint_yaml ===
    resume_path = os.path.abspath(args_cli.checkpoint)
    cfg_yaml_path = os.path.abspath(args_cli.checkpoint_yaml)
    if not os.path.exists(resume_path):
        print(f"[WARN] Checkpoint not found: {resume_path}")
        return
    if not os.path.exists(cfg_yaml_path):
        print(f"[WARN] YAML config not found: {cfg_yaml_path}")
        return
    print(f"[INFO] Loading checkpoint: {resume_path}")
    print(f"[INFO] Loading agent config (YAML): {cfg_yaml_path}")

    # ---- 创建环境 ----
    # 说明：此处仍保持原有 video 相关约定；若外层未传 video 参数则按 None 处理
    render_mode = "rgb_array" if getattr(args_cli, "video", False) else None
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=render_mode)
    if isinstance(env.unwrapped, DirectMARLEnv) and algorithm in ["ppo"]:
        env = multi_agent_to_single_agent(env)

    # step dt
    try:
        dt = env.step_dt
    except AttributeError:
        dt = env.unwrapped.step_dt


    # skrl 包装
    env = SkrlVecEnvWrapper(env, ml_framework=args_cli.ml_framework)
    base_env = _unwrap_env(env)

    # === 改动 3：用 YAML 创建 Runner，并用 checkpoint 加载 agent ===
    # 参考 skrl 官方文档：Runner.load_cfg_from_yaml(...); runner.agent.load(...)（PPO 示例）
    cfg = Runner.load_cfg_from_yaml(cfg_yaml_path)  # ← 从 agent.yaml 载入配置
    # 与原脚本一致的轻量覆盖（不改变其它逻辑）
    try:
        cfg["trainer"]["close_environment_at_exit"] = False
        cfg["agent"]["experiment"]["write_interval"] = 0
        cfg["agent"]["experiment"]["checkpoint_interval"] = 0
    except Exception:
        pass

    runner = Runner(env, cfg)
    runner.agent.load(resume_path)  # ← 加载权重
    runner.agent.set_running_mode("eval")

    # 并行环境数
    num_envs = getattr(base_env, "num_envs", getattr(getattr(base_env, "scene", None), "num_envs", 1))
    rounds = max(1, int(getattr(args_cli, "rounds", 5)))  # 若外部未提供 rounds，则沿用之前默认 5
    episodes_target = rounds * int(num_envs)

    # ---- 清零你的统计量（如果存在）----
    for name in [
        "stat_success_count",
        "stat_timeout_count",
        "stat_success_time_sum",
        "stat_timeout_time_sum",
        "stat_completed",
        "stat_avg_time",
    ]:
        if hasattr(base_env, name):
            setattr(base_env, name, 0 if "sum" not in name else 0.0)
    if hasattr(base_env, "_counted_mask"):
        try:
            base_env._counted_mask.zero_()
        except Exception:
            pass

    # 开始评测
    obs, _ = env.reset()

    # 为了保证精确跑满 n * num_envs：以底层 env 的计数为准
    def _fetch_counts():
        succ = getattr(base_env, "stat_success_count", 0)
        comp = getattr(base_env, "stat_completed", 0)
        t_succ = getattr(base_env, "stat_success_time_sum", 0.0)
        t_to = getattr(base_env, "stat_timeout_time_sum", 0.0)
        return succ, comp, t_succ, t_to

    while simulation_app.is_running():
        with torch.inference_mode():
            outputs = runner.agent.act(obs, timestep=0, timesteps=0)
            # 取确定性动作（mean）
            if hasattr(env, "possible_agents"):
                actions = {a: outputs[-1][a].get("mean_actions", outputs[0][a]) for a in env.possible_agents}
            else:
                actions = outputs[-1].get("mean_actions", outputs[0])
            obs, _, _, _, _ = env.step(actions)

        # 检查是否达到目标 episode 数
        _, comp, _, _ = _fetch_counts()
        if comp >= episodes_target:
            break

    # ---- 汇总结果（只输出你需要的三项）----
    succ, comp, t_succ, t_to = _fetch_counts()
    total_episodes = comp  # 实际完成的 episode 数
    total_time = float(t_succ + t_to)  # 仿真总用时（秒）
    avg_time = (total_time / total_episodes) if total_episodes > 0 else 0.0
    success_rate = (succ / total_episodes) if total_episodes > 0 else 0.0

    # 约定总实验次数 = rounds * num_envs（也打印实际完成 comp）
    print("\n========== Evaluation Summary ==========")
    print(f"Task: {args_cli.task}")
    print(f"Requested episodes: {episodes_target} (num_envs={num_envs}, rounds={rounds})")
    print(f"Completed episodes: {total_episodes}")
    print(f"Total sim time: {total_time:.3f} s")
    print(f"Avg sim time / episode: {avg_time:.3f} s")
    print(f"Success rate: {success_rate * 100:.2f}% ({succ}/{total_episodes})")
    print("========================================\n")


if __name__ == "__main__":
    main()
    simulation_app.close()
