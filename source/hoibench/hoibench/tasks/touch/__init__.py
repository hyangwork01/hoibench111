import gymnasium as gym

from . import agents

gym.register(
    id="Isaac-HOISmpl-Eval-Touch-v0",
    entry_point=f"{__name__}.touch_eval_env:TouchEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.touch_cfg:TouchSmplEnvCfg",
    },
)

gym.register(
    id="Isaac-HOISmpl-Train-Touch-v0",
    entry_point=f"{__name__}.touch_train_env:TouchEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.touch_cfg:TouchSmplEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-HOIH1-Eval-Touch-v0",
    entry_point=f"{__name__}.touch_eval_env:TouchEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.touch_cfg:TouchH1EnvCfg",
    },
)

gym.register(
    id="Isaac-HOIH1-Train-Touch-v0",
    entry_point=f"{__name__}.touch_train_env:TouchEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.touch_cfg:TouchH1EnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-HOIG1-Eval-Touch-v0",
    entry_point=f"{__name__}.touch_eval_env:TouchEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.touch_cfg:TouchG1EnvCfg",
    },
)

gym.register(
    id="Isaac-HOIG1-Train-Touch-v0",
    entry_point=f"{__name__}.touch_train_env:TouchEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.touch_cfg:TouchG1EnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)