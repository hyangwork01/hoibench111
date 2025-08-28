import gymnasium as gym

from . import agents

gym.register(
    id="Isaac-HOISmpl-Eval-LieBed-v0",
    entry_point=f"{__name__}.liebed_eval_env:LiebedEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.liebed_cfg:LiebedSmplEnvCfg",
    },
)

gym.register(
    id="Isaac-HOISmpl-Train-LieBed-v0",
    entry_point=f"{__name__}.liebed_train_env:LiebedEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.liebed_cfg:LiebedSmplEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-HOIH1-Eval-LieBed-v0",
    entry_point=f"{__name__}.liebed_eval_env:LiebedEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.liebed_cfg:LiebedH1EnvCfg",
    },
)

gym.register(
    id="Isaac-HOIH1-Train-LieBed-v0",
    entry_point=f"{__name__}.liebed_train_env:LiebedEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.liebed_cfg:LiebedH1EnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-HOIG1-Eval-LieBed-v0",
    entry_point=f"{__name__}.liebed_eval_env:LiebedEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.liebed_cfg:LiebedG1EnvCfg",
    },
)

gym.register(
    id="Isaac-HOIG1-Train-LieBed-v0",
    entry_point=f"{__name__}.liebed_train_env:LiebedEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.liebed_cfg:LiebedG1EnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)