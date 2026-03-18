from typing import Callable, Union

import jax
import jax.numpy as jnp
from gymnax.environments import environment

from memorax.utils.typing import Array, Key, PyTree
from gymnax.wrappers.purerl import GymnaxWrapper


class NoisyObservationWrapper(GymnaxWrapper):
    def __init__(
        self,
        env,
        mask: PyTree,
        noise_fn: Callable[[Key, tuple], Array],
    ):
        super().__init__(env)
        self.mask = mask
        self.noise_fn = noise_fn

    def _add_noise(self, key: Key, observation: PyTree) -> PyTree:
        leaves, treedef = jax.tree.flatten(observation)
        keys = jax.random.split(key, len(leaves))
        masks = jax.tree.leaves(self.mask)
        noisy_leaves = [
            leaf + self.noise_fn(k, leaf.shape) * m
            for leaf, k, m in zip(leaves, keys, masks)
        ]
        return jax.tree.unflatten(treedef, noisy_leaves)

    def reset(
        self, key: Key, params: environment.EnvParams | None = None
    ) -> tuple[Array, environment.EnvState]:
        key, noise_key = jax.random.split(key)
        observation, state = self._env.reset(key, params)
        return self._add_noise(noise_key, observation), state

    def step(
        self,
        key: Key,
        state: environment.EnvState,
        action: Union[int, float],
        params: environment.EnvParams | None = None,
    ) -> tuple[Array, environment.EnvState, float, bool, dict]:
        key, noise_key = jax.random.split(key)
        observation, state, reward, done, info = self._env.step(
            key, state, action, params
        )
        return self._add_noise(noise_key, observation), state, reward, done, info
