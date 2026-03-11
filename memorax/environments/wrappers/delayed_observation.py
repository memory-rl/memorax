from typing import Optional, Tuple, Union

import jax.numpy as jnp
from flax import struct
from gymnax.environments import environment

from memorax.utils.typing import Array, Key
from gymnax.wrappers.purerl import GymnaxWrapper


@struct.dataclass
class DelayedObservationWrapperState:
    buffer: jnp.ndarray
    env_state: environment.EnvState


class DelayedObservationWrapper(GymnaxWrapper):
    def __init__(self, env, delay: int):
        super().__init__(env)
        self.delay = delay

    def reset(
        self, key: Key, params: Optional[environment.EnvParams] = None
    ) -> Tuple[Array, DelayedObservationWrapperState]:
        obs, env_state = self._env.reset(key, params)
        buffer = jnp.zeros((self.delay,) + obs.shape)
        buffer = buffer.at[0].set(obs)
        state = DelayedObservationWrapperState(buffer=buffer, env_state=env_state)
        return jnp.zeros_like(obs), state

    def step(
        self,
        key: Key,
        state: DelayedObservationWrapperState,
        action: Union[int, float],
        params: Optional[environment.EnvParams] = None,
    ) -> Tuple[Array, DelayedObservationWrapperState, float, bool, dict]:
        obs, env_state, reward, done, info = self._env.step(
            key, state.env_state, action, params
        )
        delayed_obs = state.buffer[-1]
        buffer = jnp.roll(state.buffer, shift=1, axis=0)
        buffer = buffer.at[0].set(obs)
        state = DelayedObservationWrapperState(buffer=buffer, env_state=env_state)
        return delayed_obs, state, reward, done, info
