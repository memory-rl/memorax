from typing import Union

import jax
import jax.numpy as jnp
from gymnax.environments import environment
from gymnax.wrappers.purerl import GymnaxWrapper

from memorax.utils.typing import Array, Key, PyTree


class MaskObservationWrapper(GymnaxWrapper):
    def __init__(self, env, mask: PyTree):
        super().__init__(env)
        self.mask = mask

    def reset(
        self, key: Key, params: environment.EnvParams | None = None
    ) -> tuple[Array, environment.EnvState]:
        observation, state = self._env.reset(key, params)
        observation = jax.tree.map(lambda o, m: o * m, observation, self.mask)
        return observation, state

    def step(
        self,
        key: Key,
        state: environment.EnvState,
        action: Union[int, float],
        params: environment.EnvParams | None = None,
    ) -> tuple[Array, environment.EnvState, float, bool, dict]:
        observation, state, reward, done, info = self._env.step(
            key, state, action, params
        )
        observation = jax.tree.map(lambda o, m: o * m, observation, self.mask)
        return observation, state, reward, done, info
