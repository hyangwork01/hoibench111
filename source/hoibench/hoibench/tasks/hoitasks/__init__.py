import gymnasium as gym

from . import agents

gym.register(
    id="Isaac-HOISmpl-Direct-v0",
    entry_point=f"{__name__}.hoi_env:HOIEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.hoi_cfg:HOISmplEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-HOIH1-Direct-v0",
    entry_point=f"{__name__}.hoi_env:HOIEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.hoi_cfg:HOIH1EnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-HOIG1-Direct-v0",
    entry_point=f"{__name__}.hoi_env:HOIEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.hoi_cfg:HOIG1EnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)
