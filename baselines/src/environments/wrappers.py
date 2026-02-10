from functools import partial
from typing import Optional, Tuple, Union

import jax
import jax.numpy as jnp
from flax import struct
from gymnax.environments import environment, spaces


class GymnaxWrapper:

    def __init__(self, env):
        self._env = env

    def __getattr__(self, name):
        return getattr(self._env, name)


@struct.dataclass
class StickyActionWrapperState:
    env_state: environment.EnvState
    last_action: jnp.ndarray


class StickyActionWrapper(GymnaxWrapper):

    def __init__(self, env, sticky_prob: float = 0.0):
        super().__init__(env)
        self.sticky_prob = sticky_prob

    @partial(jax.jit, static_argnums=(0,))
    def reset(
        self, key: jax.Array, params: Optional[environment.EnvParams] = None
    ) -> Tuple[jnp.ndarray, StickyActionWrapperState]:
        obs, env_state = self._env.reset(key, params)
        state = StickyActionWrapperState(
            env_state=env_state,
            last_action=jnp.array(0, dtype=jnp.int32),
        )
        return obs, state

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        key: jax.Array,
        state: StickyActionWrapperState,
        action: Union[int, jnp.ndarray],
        params: Optional[environment.EnvParams] = None,
    ) -> Tuple[jnp.ndarray, StickyActionWrapperState, jnp.ndarray, jnp.ndarray, dict]:
        key, sticky_key = jax.random.split(key)

        use_sticky = jax.random.uniform(sticky_key) < self.sticky_prob
        effective_action = jax.lax.select(
            use_sticky,
            state.last_action,
            jnp.asarray(action, dtype=jnp.int32),
        )

        obs, env_state, reward, done, info = self._env.step(
            key, state.env_state, effective_action, params
        )

        new_state = StickyActionWrapperState(
            env_state=env_state,
            last_action=jnp.asarray(action, dtype=jnp.int32),
        )

        return obs, new_state, reward, done, info

    def observation_space(
        self, params: Optional[environment.EnvParams] = None
    ) -> spaces.Space:
        return self._env.observation_space(params)

    def action_space(
        self, params: Optional[environment.EnvParams] = None
    ) -> spaces.Space:
        return self._env.action_space(params)
