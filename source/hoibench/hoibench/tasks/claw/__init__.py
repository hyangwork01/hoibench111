import gymnasium as gym

from . import agents

gym.register(
    id="Isaac-HOISmpl-Eval-Claw-v0",
    entry_point=f"{__name__}.claw_eval_env:ClawEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.claw_cfg:ClawSmplEnvCfg",
    },
)

gym.register(
    id="Isaac-HOISmpl-Train-Claw-v0",
    entry_point=f"{__name__}.claw_train_env:ClawEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.claw_cfg:ClawSmplEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-HOIH1-Eval-Claw-v0",
    entry_point=f"{__name__}.claw_eval_env:ClawEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.claw_cfg:ClawH1EnvCfg",
    },
)

gym.register(
    id="Isaac-HOIH1-Train-Claw-v0",
    entry_point=f"{__name__}.claw_train_env:ClawEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.claw_cfg:ClawH1EnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-HOIG1-Eval-Claw-v0",
    entry_point=f"{__name__}.claw_eval_env:ClawEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.claw_cfg:ClawG1EnvCfg",
    },
)

gym.register(
    id="Isaac-HOIG1-Train-Claw-v0",
    entry_point=f"{__name__}.claw_train_env:ClawEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.claw_cfg:ClawG1EnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)