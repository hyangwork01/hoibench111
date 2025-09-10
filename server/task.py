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

DB_URL = "http://10.15.88.88:5000/api/submit"

def evaluate(file_path: str, api_token: str, robot_type: str, task: str) -> None:
    multiprocessing.set_start_method('spawn', force=True)
    processes = []
    result_queue = multiprocessing.Queue()
    results = []
    
    for variant_id in range(4):
        p = multiprocessing.Process(
            target=sub_evaluate,
            args=(file_path, api_token, robot_type, result_queue, variant_id)
        )
        processes.append(p)
        p.start()

    completed_count = 0
    start_time = time.time()
    timeout = 300
    
    while completed_count < 4 and time.time() - start_time < timeout:
        try:
            result = result_queue.get(timeout=1.0)
            results.append(result)
            completed_count += 1
            print(f"Received result from variant {len(results)-1}")
        except queue.Empty:
            for p in processes:
                if not p.is_alive():
                    p.join()
            continue
        except Exception as e:
            print(f"Error getting result: {e}")
            break
    
    for p in processes:
        if p.is_alive():
            p.terminate()  
        p.join()

    while len(results) < 4:
        results.append({'complete_rate': 0.0, 'success_time': 0.0})
    
    temp = {}
    for result in results:
        for key, value in result.items():
            temp[key] = value
    print("temp=",temp)        
    final_data = {
        "token": api_token,
        "robot_type": robot_type,
        "ckpt_path":file_path,
        "metrics":temp
    }
    try:
        response = requests.post(DB_URL, json=final_data, timeout=10)
        print(response.json())
    except Exception as e:
        print(f"Failed to submit results: {e}")

    print(f"[Task] done: {file_path} by {api_token}")


def sub_evaluate(file_path: str, api_token: str, robot_type: str,result_queue=None,variant_id=None):
    file_path = Path(file_path)

    faulthandler.enable()
    args = get_args()
    args.headless = True
    args.task = robot_type

    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    if args.nodelay:
        env_cfg.domain_rand.action_delay_view = 0
    env_cfg.env.num_envs = 100
    env_cfg.commands.resampling_time = 60
    env_cfg.rewards.is_play = True
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.max_init_terrain_level = 1
    env_cfg.terrain.border_size = 0.5

    env_cfg.depth.angle = [0, 1]
    env_cfg.noise.add_noise = True
    env_cfg.domain_rand.randomize_friction = True
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.push_interval_s = 8
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.randomize_base_com = False

    data = {}

    if variant_id == 0:
        env_cfg.env.episode_length_s = 30
        env_cfg.terrain.terrain_length = 20.
        env_cfg.terrain.terrain_width = 4.
        env_cfg.terrain.num_goals = 1
        env_cfg.terrain.proportions = [("single", 12, 0.5)]
    if variant_id == 1:
        env_cfg.env.episode_length_s = 15
        env_cfg.terrain.terrain_length = 7.
        env_cfg.terrain.terrain_width = 4.
        env_cfg.terrain.num_goals = 1
        env_cfg.terrain.proportions = [("single", 13, 0.5)]
    if variant_id == 2:
        env_cfg.env.episode_length_s = 30
        env_cfg.terrain.terrain_length = 20.
        env_cfg.terrain.terrain_width = 4.
        env_cfg.terrain.num_goals = 20
        env_cfg.terrain.proportions = [("single", 14, 0.5)]
    if variant_id == 3:
        env_cfg.env.episode_length_s = 40
        env_cfg.terrain.terrain_length = 30.
        env_cfg.terrain.terrain_width = 4.
        env_cfg.terrain.num_goals = 30
        env_cfg.terrain.proportions = [("single", 15, 0.5)]   
    
    env: HumanoidRobot
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    policy = torch.jit.load(file_path, map_location=env.device)
    obs = env.get_observations()

    actions = torch.zeros(env.num_envs, 12, device=env.device, requires_grad=False)
    for i in range(10*int(env.max_episode_length)):
        obs = obs[:,:175]
        actions = policy(obs.detach())
        obs, _, rews, dones, infos = env.step(actions.detach())
        times = env.total_times
        if (times >= 100):
            success_time = (env.success_times / env.total_times)/env.max_episode_length
            complete_rate =(env.complete_times / env.total_times)
            complete_rate = max(0.0, min(1.0, complete_rate))
            success_time = max(0.0, min(1.0, success_time))

            # data['complete_rate' + str(variant_id)] = complete_rate
            # data['success_time' + str(variant_id)] = success_time
            data['score' + str(variant_id)] = complete_rate*0.9 + (1-success_time)*0.1
            break
    if result_queue:
            result_queue.put(data)
            
    return data