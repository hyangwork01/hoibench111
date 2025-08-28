# Humanoid–Object Interaction Challenge Template

## Overview

This repository provides the code template for the ICCV 2025 workshop **Humanoid–Object Interaction Challenge**, used to train and evaluate humanoid robots’ interaction capabilities in multi-object scenes.

## Task Definitions

The six benchmark tasks provide both training (Train) and evaluation (Eval) environments, and support multiple robot models including **SMPL**, **Unitree H1**, and **Unitree G1**.

* **CarryBox**: Lift a box from the ground and carry it along the +X direction for a certain distance. Success requires the box to rise by at least `lift_height` (default 0.15 m) and the XY error to be less than `success_xy_threshold` (default 0.25 m).
* **Claw**: Toys and a target position are randomly placed on a tabletop. The robot must push the toy to the target position and hold it there for `eval_hold_success_steps` consecutive frames (default 5 frames).
* **LieBed**: Lie down at the center of the bed; must satisfy vertical height difference < `lie_z_threshold` (default 0.05 m), XY distance < `lie_xy_threshold` (default 0.35 m), and velocity < `lie_vel_threshold`, and maintain these conditions for several frames.
* **PushBox**: Push the box so that its displacement exceeds `push_target_offset` (default 0.5 m).
* **SitChair**: Sit down at the center of the chair; success is determined by a combination of seat height difference, XY error, and velocity thresholds.
* **Touch**: Use a specified body part to touch a randomly generated target point; success requires the distance to be less than `touch_threshold` (default 0.03 m) and to execute at least `min_success_steps` steps.

## Observations

All tasks’ observations consist of two parts:

1. **Robot proprioception**: joint positions/velocities, base/root height and 6D rotation, linear velocity, angular velocity, etc.
2. **Interaction information**: task-related relative object positions, OBB/AABB sizes, poses, or target vectors, etc.

## Installation

1. Follow the [Isaac Lab Installation Guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/pip_installation.html) to set up the runtime environment.
2. Clone and install this template in editable mode:

   ```bash
   python -m pip install -e source/hoibench
   ```

## Training (Train)

List available environment names, then start training:

```bash
python scripts/list_envs.py
python scripts/skrl/train.py \
  --task=Isaac-HOISmpl-Train-Claw-v0 \
  --num_envs=16 \
  --algorithm=PPO
```



## Evaluation (Eval)

After training, run batch evaluation. Provide the checkpoint and the corresponding YAML config:

```bash
python scripts/skrl/eval.py \
  --task=Isaac-HOIG1-Eval-Claw-v0 \
  --num_envs=16 \
  --algorithm=PPO \
  --checkpoint=logs/skrl/humanoid_direct/2025-08-28_12-40-09_ppo_torch/checkpoints/best_agent.pt \
  --checkpoint_yaml=logs/skrl/humanoid_direct/2025-08-28_12-40-09_ppo_torch/params/agent.yaml

```

The evaluation process reports the number of successes, timeouts, and the average completion time, and performs a global reset after all environments finish.

## Metrics

Each evaluation environment maintains the following statistics:

* `stat_success_count`: number of successes
* `stat_timeout_count`: number of timeouts
* `stat_avg_time`: average completion time (see the Claw evaluation environment source code for an example)

## Environment Debugging

Run the zero-action or random-action scripts first to quickly verify that the environment is configured correctly:

```bash
python scripts/zero_agent.py --task=<TASK_NAME>
```

```bash
python scripts/random_agent.py --task=<TASK_NAME>
```
