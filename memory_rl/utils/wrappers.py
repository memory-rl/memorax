from functools import partial
from typing import Any, Optional, Tuple, Union

import chex
import jax
import jax.numpy as jnp
from gymnax.environments import environment, spaces


class GymnaxWrapper(object):
    """Base class for Gymnax wrappers."""

    def __init__(self, env):
        self._env = env

    def __getattr__(self, name):
        return getattr(self._env, name)


# Implementation from https://github.com/taodav/pobax/blob/main/pobax/envs/wrappers
class MaskObservationWrapper(GymnaxWrapper):
    def __init__(self, env, mask_dims: list, **kwargs):
        super().__init__(env)
        self.mask_dims = jnp.array(mask_dims, dtype=int)

    def observation_space(self, params) -> spaces.Box:
        assert isinstance(
            self._env.observation_space(params), spaces.Box
        ), "Only Box spaces are supported for now."
        low = self._env.observation_space(params).low
        if isinstance(low, jnp.ndarray):
            low = low[self.mask_dims]

        high = self._env.observation_space(params).high
        if isinstance(high, jnp.ndarray):
            high = high[self.mask_dims]

        return spaces.Box(
            low=low,
            high=high,
            shape=(self.mask_dims.shape[0],),
            dtype=self._env.observation_space(params).dtype,
        )

    @partial(jax.jit, static_argnums=(0, -1))
    def reset(
        self, key: chex.PRNGKey, params: Optional[environment.EnvParams] = None
    ) -> Tuple[chex.Array, environment.EnvState]:
        obs, state = self._env.reset(key, params)
        obs = obs[self.mask_dims]
        return obs, state

    @partial(jax.jit, static_argnums=(0, -1))
    def step(
        self,
        key: chex.PRNGKey,
        state: environment.EnvState,
        action: Union[int, float],
        params: Optional[environment.EnvParams] = None,
    ) -> Tuple[chex.Array, environment.EnvState, float, bool, dict]:
        obs, state, reward, done, info = self._env.step(key, state, action, params)
        obs = obs[self.mask_dims]
        return obs, state, reward, done, info
