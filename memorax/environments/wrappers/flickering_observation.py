from typing import Callable, Optional, Tuple, Union

import jax
import jax.numpy as jnp
from gymnax.environments import environment

from memorax.utils.typing import Array, Key
from gymnax.wrappers.purerl import GymnaxWrapper


class FlickeringObservationWrapper(GymnaxWrapper):
    def __init__(
        self,
        env,
        p: float,
        fill_fn: Callable[[Key, tuple], Array] = lambda key, shape: jnp.zeros(shape),
    ):
        super().__init__(env)
        self.p = p
        self.fill_fn = fill_fn

    def step(
        self,
        key: Key,
        state: environment.EnvState,
        action: Union[int, float],
        params: Optional[environment.EnvParams] = None,
    ) -> Tuple[Array, environment.EnvState, float, bool, dict]:
        key, flicker_key, fill_key = jax.random.split(key, 3)
        obs, state, reward, done, info = self._env.step(key, state, action, params)
        visible = jax.random.uniform(flicker_key) >= self.p
        fill = self.fill_fn(fill_key, obs.shape)
        obs = jnp.where(visible, obs, fill)
        return obs, state, reward, done, info
