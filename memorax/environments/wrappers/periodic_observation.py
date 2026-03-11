from typing import Callable, Optional, Tuple, Union

import jax
import jax.numpy as jnp
from flax import struct
from gymnax.environments import environment

from memorax.utils.typing import Array, Key
from gymnax.wrappers.purerl import GymnaxWrapper


@struct.dataclass
class PeriodicObservationWrapperState:
    step: int
    env_state: environment.EnvState


class PeriodicObservationWrapper(GymnaxWrapper):
    def __init__(
        self,
        env,
        period: int,
        fill_fn: Callable[[Key, tuple], Array] = lambda key, shape: jnp.zeros(shape),
    ):
        super().__init__(env)
        self.period = period
        self.fill_fn = fill_fn

    def reset(
        self, key: Key, params: Optional[environment.EnvParams] = None
    ) -> Tuple[Array, PeriodicObservationWrapperState]:
        obs, state = self._env.reset(key, params)
        state = PeriodicObservationWrapperState(step=0, env_state=state)
        return obs, state

    def step(
        self,
        key: Key,
        state: PeriodicObservationWrapperState,
        action: Union[int, float],
        params: Optional[environment.EnvParams] = None,
    ) -> Tuple[Array, PeriodicObservationWrapperState, float, bool, dict]:
        key, fill_key = jax.random.split(key)
        obs, env_state, reward, done, info = self._env.step(
            key, state.env_state, action, params
        )
        new_step = state.step + 1
        visible = new_step % self.period == 0
        fill = self.fill_fn(fill_key, obs.shape)
        obs = jnp.where(visible, obs, fill)
        state = PeriodicObservationWrapperState(step=new_step, env_state=env_state)
        return obs, state, reward, done, info
