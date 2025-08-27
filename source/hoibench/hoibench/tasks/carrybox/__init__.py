import gymnasium as gym

from . import agents


# SMPL humanoid
gym.register(
    id="Isaac-HOISmpl-Eval-CarryBox-v0",
    entry_point=f"{__name__}.carrybox_eval_env:CarryboxEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.carrybox_cfg:CarryboxSmplEnvCfg",
    },
)

gym.register(
    id="Isaac-HOISmpl-Train-CarryBox-v0",
    entry_point=f"{__name__}.carrybox_train_env:CarryboxEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.carrybox_cfg:CarryboxSmplEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

# Unitree H1
gym.register(
    id="Isaac-HOIH1-Eval-CarryBox-v0",
    entry_point=f"{__name__}.carrybox_eval_env:CarryboxEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.carrybox_cfg:CarryboxH1EnvCfg",
    },
)

gym.register(
    id="Isaac-HOIH1-Train-CarryBox-v0",
    entry_point=f"{__name__}.carrybox_train_env:CarryboxEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.carrybox_cfg:CarryboxH1EnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

# Agility G1
gym.register(
    id="Isaac-HOIG1-Eval-CarryBox-v0",
    entry_point=f"{__name__}.carrybox_eval_env:CarryboxEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.carrybox_cfg:CarryboxG1EnvCfg",
    },
)

gym.register(
    id="Isaac-HOIG1-Train-CarryBox-v0",
    entry_point=f"{__name__}.carrybox_train_env:CarryboxEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.carrybox_cfg:CarryboxG1EnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)
