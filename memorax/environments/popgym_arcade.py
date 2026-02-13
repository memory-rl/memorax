from typing import Any

import jax.numpy as jnp
from flax import struct
from gymnax.environments import spaces

from memorax.utils.wrappers import GymnaxWrapper


@struct.dataclass(frozen=True)
class EnvParams:
    env_params: Any
    max_steps_in_episode: int


class PopGymArcadeWrapper(GymnaxWrapper):
    def reset(self, key, params):
        return self._env.reset(key, params.env_params)

    def step(self, key, state, action, params):
        return self._env.step(key, state, action, params.env_params)


class FlattenObservationWrapper(GymnaxWrapper):
    def __init__(self, env):
        super().__init__(env)
        raw_shape = env.observation_space(None).shape
        self._flat_dim = int(jnp.prod(jnp.array(raw_shape)))
        self._raw_ndim = len(raw_shape)

    def _flatten(self, obs):
        return obs.reshape(obs.shape[:-self._raw_ndim] + (self._flat_dim,))

    def reset(self, key, params):
        obs, state = self._env.reset(key, params)
        return self._flatten(obs), state

    def step(self, key, state, action, params):
        obs, state, reward, done, info = self._env.step(key, state, action, params)
        return self._flatten(obs), state, reward, done, info

    def observation_space(self, params):
        return spaces.Box(
            low=0.0, high=1.0, shape=(self._flat_dim,),
            dtype=self._env.observation_space(params).dtype,
        )


def make(env_id, flatten_obs=False, **kwargs):
    import popgym_arcade

    env, env_params = popgym_arcade.make(env_id, **kwargs)
    env = PopGymArcadeWrapper(env)

    if flatten_obs:
        env = FlattenObservationWrapper(env)

    env_params = EnvParams(
        env_params=env_params, max_steps_in_episode=env._env.max_steps_in_episode
    )
    return env, env_params
