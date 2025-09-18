from __future__ import annotations

from pathlib import Path
import time
# from legged_gym.envs import *
# from legged_gym.utils import get_args, task_registry
# from terrain_base.config import terrain_config
import gymnasium as gym

import requests
import torch
import faulthandler
import multiprocessing
import queue


import os, sys, torch, gymnasium as gym
from pathlib import Path
from packaging import version

from isaaclab.app import AppLauncher

# import hoibench.tasks  # noqa: F401  # 如果你有自定义任务
import skrl


from skrl.utils.runner.torch import Runner

SKRL_VERSION = "1.4.2"
if version.parse(skrl.__version__) < version.parse(SKRL_VERSION):
    raise RuntimeError(f"Unsupported skrl version: {skrl.__version__}. Need >= {SKRL_VERSION}")


DB_URL = "http://10.15.88.88:5555/api/submit"


def get_isaac_app_and_cfg():
    # —— App 启动相关（逐行声明，便于将来改动）——
    HEADLESS         = True       # 无界面
    LIVESTREAM       = 0          # 0=关; 1/2=不同流式方案
    ENABLE_CAMERAS   = False      # 任务包含相机时改 True
    DEVICE           = "cuda:0"   # "cpu" / "cuda" / "cuda:N"

    # —— 评估运行相关（逐行声明，后续作为“运行配置”使用）——
    NUM_ENVS         = 1          # env 个数（None 表示沿用任务默认）
    ML_FRAMEWORK     = "torch"    # "torch" / "jax" / "jax-numpy"
    ALGORITHM        = "PPO"      # 用于决定是否做 MARL->单体转换等
    ROUNDS           = 2          # 每个 env 跑的 episode 数
    # SEED             = None       # 需要可复现实验再设置
    SEED             = 0       # 需要可复现实验再设置

    app_kwargs = dict(
        headless=HEADLESS,
        livestream=LIVESTREAM,
        enable_cameras=ENABLE_CAMERAS,
        device=DEVICE,
        # experience=""  # 留空即可；需要自定义 .kit 时再设置
    )
    run_cfg = dict(
        num_envs=NUM_ENVS,
        device=DEVICE,
        ml_framework=ML_FRAMEWORK,
        algorithm=ALGORITHM,
        rounds=ROUNDS,
        seed=SEED,
    )

    app_launcher = AppLauncher(**app_kwargs)
    return app_launcher.app, run_cfg


def _unwrap_env(env):
    base = getattr(env, "unwrapped", env)
    seen = set()
    while hasattr(base, "unwrapped") and id(base.unwrapped) not in seen:
        seen.add(id(base))
        if base.unwrapped is base:
            break
        base = base.unwrapped
    return base

def print_payload_block(payload: dict) -> None:
    sep = "=" * 50
    print(sep)

    # 顶层字段
    for key in ["token", "robot_type", "task", "env_id", "ckpt_path", "yaml_path"]:
        if key in payload:
            print(f"{key}：{payload[key]}")   # f-string 内插值与格式控制，见官方文档

    # metrics 子项
    metrics = payload.get("metrics", {})
    print("metrics:")
    def _fmt(v):
        return f"{v:.4f}" if isinstance(v, (int, float)) else v
    for mkey in ["success_rate", "avg_time", "score"]:
        if mkey in metrics:
            print(f"    {mkey}：{_fmt(metrics[mkey])}")

    print(sep)



def evaluate(ckpt_path: str, yaml_path: str, api_token: str, robot_type: str, task: str) -> None:
    out_task = task
    match task.lower():  # Python 3.10+
        case "lie":
            task = "LieBed"
        case "push":
            task = "PushBox"
        case "touch":
            task = "Touch"
        case "lift":
            task = "CarryBox"
        case "sit":
            task = "SitChair"
        case "claw":
            task = "Claw"
        case _:
            raise ValueError(f"unknown task: {task}")

    match robot_type:
        case "g1":
            robot_type = "G1"
        case "h1":
            robot_type = "H1"
        case "smpl":
            robot_type = "Smpl"
        case _:
            raise ValueError(f"unknown task: {task}")        

    task_name = str(f"Isaac-HOI{robot_type}-Eval-{task}-v0")

    simulation_app, run_cfg = get_isaac_app_and_cfg()
    from isaaclab.envs import (
        DirectMARLEnv, DirectMARLEnvCfg, DirectRLEnvCfg,
        ManagerBasedRLEnvCfg, multi_agent_to_single_agent,
    )
    from isaaclab_tasks.utils.hydra import hydra_task_config
    from isaaclab_rl.skrl import SkrlVecEnvWrapper
    import hoibench.tasks  # noqa: F401

    # import time

    # print("开始")
    # print(task_name)
    # print(task_name.split(":")[-1])
    # time.sleep(5)   # 程序暂停 3 秒
    # print("3 秒后输出")


    # "g1", "h1", "smpl"
    # "Lie", "Push", "Touch","Lift","Sit","Claw"
    # Isaac-HOISmpl-Eval-SitChair-v0 \Isaac-HOIG1-Eval-SitChair-v0
    @hydra_task_config(task_name, agent_cfg_entry_point=None)
    def _run_once(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, experiment_cfg: dict):
        env_cfg.scene.num_envs = int(run_cfg["num_envs"])
        env_cfg.sim.device = run_cfg["device"]
        skrl.config.jax.backend = "numpy"
        algorithm = run_cfg["algorithm"].lower()
        resume_path = os.path.abspath(ckpt_path)
        cfg_yaml_path = os.path.abspath(yaml_path)
        if not os.path.exists(resume_path):
            print(f"[WARN] Checkpoint not found: {resume_path}")
            return
        if not os.path.exists(cfg_yaml_path):
            print(f"[WARN] YAML config not found: {cfg_yaml_path}")
            return
        print(f"[INFO] Loading checkpoint: {resume_path}")
        print(f"[INFO] Loading agent config (YAML): {cfg_yaml_path}")

        render_mode = "rgb_array"
        env = gym.make(task_name, cfg=env_cfg, render_mode=render_mode)

        try:
            dt = env.step_dt
        except AttributeError:
            dt = env.unwrapped.step_dt
        
        env = SkrlVecEnvWrapper(env, ml_framework=run_cfg["ml_framework"])
        base_env = _unwrap_env(env)
        cfg = Runner.load_cfg_from_yaml(cfg_yaml_path)        

        try:
            cfg["trainer"]["close_environment_at_exit"] = False
            cfg["agent"]["experiment"]["write_interval"] = 0
            cfg["agent"]["experiment"]["checkpoint_interval"] = 0
        except Exception:
            pass

        runner = Runner(env, cfg)
        runner.agent.load(resume_path)          
        runner.agent.set_running_mode("eval")

            
        num_envs = base_env.num_envs
        rounds = run_cfg["rounds"]               
        episodes_target = rounds * int(num_envs)

        obs, _ = env.reset()

                                            
        def _fetch_counts():
            succ = base_env.stat_success_count
            comp = base_env.stat_completed
            t_succ = base_env.stat_success_time_sum
            t_to = base_env.stat_timeout_time_sum
            return succ, comp, t_succ, t_to

        while simulation_app.is_running():
            with torch.inference_mode():
                outputs = runner.agent.act(obs, timestep=0, timesteps=0)
                            
                if hasattr(env, "possible_agents"):
                    actions = {a: outputs[-1][a].get("mean_actions", outputs[0][a]) for a in env.possible_agents}
                else:
                    actions = outputs[-1].get("mean_actions", outputs[0])
                obs, _, _, _, _ = env.step(actions)

                                
            _, comp, _, _ = _fetch_counts()
            if comp >= episodes_target:
                break

                                
        succ, comp, t_succ, t_to = _fetch_counts()
        total_episodes = comp                   
        total_time = float(t_succ + t_to)            
        avg_time = (total_time / total_episodes)
        success_rate = (succ / total_episodes)

        print("\n========== Evaluation Summary ==========")
        print(f"Task: {task_name}")
        print(f"Requested episodes: {episodes_target} (num_envs={num_envs}, rounds={rounds})")
        print(f"Completed episodes: {total_episodes}")
        print(f"Total sim time: {total_time:.3f} s")
        print(f"Avg sim time / episode: {avg_time:.3f} s")
        print(f"Success rate: {success_rate * 100:.2f}% ({succ}/{total_episodes})")
        print("========================================\n")

        score = success_rate * 0.9 + (1.0 - avg_time / base_env.max_episode_length_s) *0.1
        metrics = {
            "success_rate": success_rate,
            "avg_time": avg_time,
            "score": score,
        }

        payload = {
            "token": api_token,
            "robot_type": robot_type.lower(),
            "task": out_task,
            "env_id": task_name,         
            "ckpt_path": str(ckpt_path),
            "yaml_path": str(yaml_path),
            "metrics": metrics,
        }
        print_payload_block(payload)
        try:
            response = requests.post(DB_URL, json=payload, timeout=10)
            print(response.json())
        except Exception as e:
            print(f"Failed to submit results: {e}")

        print(f"[Task] Eval {task_name} done: {ckpt_path} by {api_token}")
        base_env.close()
        simulation_app.close()
        return
    
    _run_once()

