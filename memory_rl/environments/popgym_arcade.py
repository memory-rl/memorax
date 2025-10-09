from typing import Any
from dataclasses import dataclass

import popgym_arcade

from memory_rl.utils.wrappers import GymnaxWrapper


@dataclass(frozen=True)
class EnvParams:
    env_params: Any
    max_steps_in_episode: int


class PopGymArcadeWrapper(GymnaxWrapper):

    def reset(self, key, params):
        return self._env.reset_env(key, params.env_params)

    def step(self, key, state, action, params):
        return self._env.step_env(key, state, action, params.env_params)


def make(cfg):
    kwargs = cfg.kwargs or {}
    env, env_params = popgym_arcade.make(cfg.env_id, **kwargs)
    env = PopGymArcadeWrapper(env)
    env_params = EnvParams(
        env_params=env_params, max_steps_in_episode=env.max_steps_in_episode
    )
    return env, env_params
