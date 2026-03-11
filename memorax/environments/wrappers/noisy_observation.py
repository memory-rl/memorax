from typing import Callable, Optional, Tuple, Union

import jax
import jax.numpy as jnp
from gymnax.environments import environment

from memorax.utils.typing import Array, Key
from gymnax.wrappers.purerl import GymnaxWrapper


class NoisyObservationWrapper(GymnaxWrapper):
    def __init__(
        self,
        env,
        noise_dims: list,
        noise_fn: Callable[[Key, tuple], Array],
    ):
        super().__init__(env)
        self.noise_dims = jnp.array(noise_dims, dtype=int)
        self.noise_fn = noise_fn

    def _add_noise(self, key: Key, obs: Array) -> Array:
        noise = jnp.zeros_like(obs)
        sampled = self.noise_fn(key, self.noise_dims.shape)
        noise = noise.at[self.noise_dims].set(sampled)
        return obs + noise

    def reset(
        self, key: Key, params: Optional[environment.EnvParams] = None
    ) -> Tuple[Array, environment.EnvState]:
        key, noise_key = jax.random.split(key)
        obs, state = self._env.reset(key, params)
        return self._add_noise(noise_key, obs), state

    def step(
        self,
        key: Key,
        state: environment.EnvState,
        action: Union[int, float],
        params: Optional[environment.EnvParams] = None,
    ) -> Tuple[Array, environment.EnvState, float, bool, dict]:
        key, noise_key = jax.random.split(key)
        obs, state, reward, done, info = self._env.step(key, state, action, params)
        return self._add_noise(noise_key, obs), state, reward, done, info
