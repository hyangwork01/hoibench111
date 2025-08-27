import gymnasium as gym

from . import agents

gym.register(
    id="Isaac-HOISmpl-Eval-SitChair-v0",
    entry_point=f"{__name__}.sitchair_eval_env:SitchairEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.sitchair_cfg:SitchairSmplEnvCfg",
    },
)

gym.register(
    id="Isaac-HOISmpl-Train-SitChair-v0",
    entry_point=f"{__name__}.sitchair_train_env:SitchairEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.sitchair_cfg:SitchairSmplEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-HOIH1-Eval-SitChair-v0",
    entry_point=f"{__name__}.sitchair_eval_env:SitchairEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.sitchair_cfg:SitchairH1EnvCfg",
    },
)

gym.register(
    id="Isaac-HOIH1-Train-SitChair-v0",
    entry_point=f"{__name__}.sitchair_train_env:SitchairEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.sitchair_cfg:SitchairH1EnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-HOIG1-Eval-SitChair-v0",
    entry_point=f"{__name__}.sitchair_eval_env:SitchairEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.sitchair_cfg:SitchairG1EnvCfg",
    },
)

gym.register(
    id="Isaac-HOIG1-Train-SitChair-v0",
    entry_point=f"{__name__}.sitchair_train_env:SitchairEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.sitchair_cfg:SitchairG1EnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)