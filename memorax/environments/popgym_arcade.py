from typing import Any

from flax import struct

from memorax.utils.wrappers import GymnaxWrapper


@struct.dataclass(frozen=True)
class EnvParams:
    env_params: Any
    max_steps_in_episode: int


class PopGymArcadeWrapper(GymnaxWrapper):

    def reset(self, key, params):
        return self._env.reset_env(key, params.env_params)

    def step(self, key, state, action, params):
        return self._env.step_env(key, state, action, params.env_params)


def make(env_id, **kwargs):
    import popgym_arcade

    env, env_params = popgym_arcade.make(env_id, **kwargs)
    env = PopGymArcadeWrapper(env)
    env_params = EnvParams(
        env_params=env_params, max_steps_in_episode=env.max_steps_in_episode
    )
    return env, env_params
