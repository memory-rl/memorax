from typing import Callable, Union

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
        params: environment.EnvParams | None = None,
    ) -> tuple[Array, environment.EnvState, float, bool, dict]:
        key, flicker_key, fill_key = jax.random.split(key, 3)
        observation, state, reward, done, info = self._env.step(
            key, state, action, params
        )
        visible = jax.random.uniform(flicker_key) >= self.p
        observation = jax.tree.map(
            lambda o: jnp.where(visible, o, self.fill_fn(fill_key, o.shape)),
            observation,
        )
        return observation, state, reward, done, info
