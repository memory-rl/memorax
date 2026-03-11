from typing import Optional, Tuple, Union

import jax.numpy as jnp
from flax import struct
from gymnax.environments import environment

from memorax.utils.typing import Array, Key
from gymnax.wrappers.purerl import GymnaxWrapper


@struct.dataclass
class NormalizeObservationWrapperState:
    mean: jnp.ndarray
    var: jnp.ndarray
    count: float
    env_state: environment.EnvState


class NormalizeObservationWrapper(GymnaxWrapper):
    def __init__(self, env, eps: float = 1e-8):
        super().__init__(env)
        self.eps = eps

    def _welford_update(self, mean, var, count, obs):
        count = count + 1
        delta = obs - mean
        mean = mean + delta / count
        delta2 = obs - mean
        var = var + (delta * delta2 - var) / count
        return mean, var, count

    def reset(
        self, key: Key, params: Optional[environment.EnvParams] = None
    ) -> Tuple[Array, NormalizeObservationWrapperState]:
        obs, env_state = self._env.reset(key, params)
        mean = jnp.zeros_like(obs)
        var = jnp.ones_like(obs)
        count = self.eps
        mean, var, count = self._welford_update(mean, var, count, obs)
        state = NormalizeObservationWrapperState(
            mean=mean,
            var=var,
            count=count,
            env_state=env_state,
        )
        return (obs - mean) / jnp.sqrt(var + self.eps), state

    def step(
        self,
        key: Key,
        state: NormalizeObservationWrapperState,
        action: Union[int, float],
        params: Optional[environment.EnvParams] = None,
    ) -> Tuple[Array, NormalizeObservationWrapperState, float, bool, dict]:
        obs, env_state, reward, done, info = self._env.step(
            key, state.env_state, action, params
        )
        mean, var, count = self._welford_update(
            state.mean, state.var, state.count, obs
        )
        state = NormalizeObservationWrapperState(
            mean=mean,
            var=var,
            count=count,
            env_state=env_state,
        )
        return (
            (obs - state.mean) / jnp.sqrt(state.var + self.eps),
            state,
            reward,
            done,
            info,
        )
