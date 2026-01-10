from typing import Any

from flax import struct

@struct.dataclass(frozen=True)
class EnvParams:
    env_params: Any
    max_steps_in_episode

def make(env_id, **kwargs):
    import popgym_arcade

    env, env_params = popgym_arcade.make(env_id, **kwargs)
    env_params = EnvParams(
        env_params=env_params, max_steps_in_episode=env.max_steps_in_episode
    )
    return env, env_params
