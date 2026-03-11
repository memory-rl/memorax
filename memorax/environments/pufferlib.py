from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np
from flax import struct
from gymnax.environments import spaces

from memorax.utils.typing import Array, Key


@struct.dataclass
class PufferLibEnvState:
    step: int = 0


class PufferLibWrapper:

    def __init__(self, env):
        self._env = env
        self.num_envs = env.num_envs

        obs_space = env.single_observation_space
        self.obs_shape = obs_space.shape
        self.obs_dtype = jnp.dtype(obs_space.dtype)

        self.num_actions = env.single_action_space.n

    @property
    def default_params(self):
        return None

    def reset(self, key: Key, params=None) -> Tuple[Array, PufferLibEnvState]:

        def _reset(key):
            obs, _ = self._env.reset()
            return jnp.array(obs, dtype=self.obs_dtype)

        obs = jax.pure_callback(
            _reset,
            jax.ShapeDtypeStruct(self.obs_shape, self.obs_dtype),
            key,
            vmap_method="broadcast_all",
        )

        state = PufferLibEnvState(step=0)
        return obs, state

    def step(
        self,
        key: Key,
        state: PufferLibEnvState,
        action: Array,
        params=None,
    ) -> Tuple[Array, PufferLibEnvState, Array, Array, dict]:

        def _step(action):
            action = np.asarray(action, dtype=np.int32)
            obs, rewards, dones, truncs, infos = self._env.step(action)

            return (
                jnp.array(obs, dtype=self.obs_dtype),
                jnp.array(rewards, dtype=jnp.float32),
                jnp.array(dones | truncs, dtype=jnp.bool_),
            )

        obs, rewards, dones = jax.pure_callback(
            _step,
            (
                jax.ShapeDtypeStruct(self.obs_shape, self.obs_dtype),
                jax.ShapeDtypeStruct((), jnp.float32),
                jax.ShapeDtypeStruct((), jnp.bool_),
            ),
            action,
            vmap_method="broadcast_all",
        )

        new_state = PufferLibEnvState(step=state.step + 1)
        return obs, new_state, rewards, dones, {}

    def observation_space(self, params=None) -> spaces.Box:
        return spaces.Box(
            low=-jnp.inf,
            high=jnp.inf,
            shape=self.obs_shape,
            dtype=self.obs_dtype,
        )

    def action_space(self, params=None) -> spaces.Discrete:
        return spaces.Discrete(self.num_actions)
