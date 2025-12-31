from functools import partial
from typing import Optional, Tuple, Union

import jax
import jax.numpy as jnp
from flax import struct
from gymnax.environments import environment, spaces

from memorax.utils.typing import Array, Key


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
        self, key: Key, params: Optional[environment.EnvParams] = None
    ) -> Tuple[Array, environment.EnvState]:
        obs, state = self._env.reset(key, params)
        obs = obs[self.mask_dims]
        return obs, state

    @partial(jax.jit, static_argnums=(0, -1))
    def step(
        self,
        key: Key,
        state: environment.EnvState,
        action: Union[int, float],
        params: Optional[environment.EnvParams] = None,
    ) -> Tuple[Array, environment.EnvState, float, bool, dict]:
        obs, state, reward, done, info = self._env.step(key, state, action, params)
        obs = obs[self.mask_dims]
        return obs, state, reward, done, info


@struct.dataclass
class NormalizeObservationWrapperState:
    mean: jnp.ndarray
    var: jnp.ndarray
    count: float
    env_state: environment.EnvState


class NormalizeObservationWrapper(GymnaxWrapper):
    def __init__(self, env):
        super().__init__(env)

    def reset(self, key, params=None):
        obs, state = self._env.reset(key, params)
        state = NormalizeObservationWrapperState(
            mean=jnp.zeros_like(obs),
            var=jnp.ones_like(obs),
            count=1e-4,
            env_state=state,
        )
        batch_mean = jnp.mean(obs, axis=0)
        batch_var = jnp.var(obs, axis=0)
        batch_count = obs.shape[0]

        delta = batch_mean - state.mean
        tot_count = state.count + batch_count

        new_mean = state.mean + delta * batch_count / tot_count
        m_a = state.var * state.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + jnp.square(delta) * state.count * batch_count / tot_count
        new_var = M2 / tot_count
        new_count = tot_count

        state = NormalizeObservationWrapperState(
            mean=new_mean,
            var=new_var,
            count=new_count,
            env_state=state.env_state,
        )

        return (obs - state.mean) / jnp.sqrt(state.var + 1e-8), state

    def step(self, key, state, action, params=None):
        obs, env_state, reward, done, info = self._env.step(
            key, state.env_state, action, params
        )

        batch_mean = jnp.mean(obs, axis=0)
        batch_var = jnp.var(obs, axis=0)
        batch_count = obs.shape[0]

        delta = batch_mean - state.mean
        tot_count = state.count + batch_count

        new_mean = state.mean + delta * batch_count / tot_count
        m_a = state.var * state.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + jnp.square(delta) * state.count * batch_count / tot_count
        new_var = M2 / tot_count
        new_count = tot_count

        state = NormalizeObservationWrapperState(
            mean=new_mean,
            var=new_var,
            count=new_count,
            env_state=env_state,
        )
        return (
            (obs - state.mean) / jnp.sqrt(state.var + 1e-8),
            state,
            reward,
            done,
            info,
        )

class ClipActionWrapper(GymnaxWrapper):
    def __init__(self, env, low=-1.0, high=1.0):
        super().__init__(env)
        self.low = low
        self.high = high

    def step(self, key, state, action, params=None):
        action = jnp.clip(action, self.low, self.high)
        return self._env.step(key, state, action, params)


class ScaleRewardWrapper(GymnaxWrapper):
    def __init__(self, env, scale=1.0):
        super().__init__(env)
        self.scale = scale

    def step(self, key, state, action, params=None):
        obs, env_state, reward, done, info = self._env.step(key, state, action, params)
        return obs, env_state, self.scale * reward, done, info
