import gymnasium as gym

from . import agents


# SMPL humanoid
gym.register(
    id="Isaac-HOISmpl-Eval-PushBox-v0",
    entry_point=f"{__name__}.pushbox_eval_env:PushboxEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pushbox_cfg:PushboxSmplEnvCfg",
    },
)

gym.register(
    id="Isaac-HOISmpl-Train-PushBox-v0",
    entry_point=f"{__name__}.pushbox_train_env:PushboxEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pushbox_cfg:PushboxSmplEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

# Unitree H1
gym.register(
    id="Isaac-HOIH1-Eval-PushBox-v0",
    entry_point=f"{__name__}.pushbox_eval_env:PushboxEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pushbox_cfg:PushboxH1EnvCfg",
    },
)

gym.register(
    id="Isaac-HOIH1-Train-PushBox-v0",
    entry_point=f"{__name__}.pushbox_train_env:PushboxEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pushbox_cfg:PushboxH1EnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

# Agility G1
gym.register(
    id="Isaac-HOIG1-Eval-PushBox-v0",
    entry_point=f"{__name__}.pushbox_eval_env:PushboxEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pushbox_cfg:PushboxG1EnvCfg",
    },
)

gym.register(
    id="Isaac-HOIG1-Train-PushBox-v0",
    entry_point=f"{__name__}.pushbox_train_env:PushboxEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pushbox_cfg:PushboxG1EnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)
