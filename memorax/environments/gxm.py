from typing import Any, Optional

import jax.numpy as jnp
from gymnax.environments import spaces

from memorax.utils.typing import Array, EnvParams, Key
from memorax.utils.wrappers import GymnaxWrapper


# Copied from https://github.com/huterguier/gxm/blob/dev/gxm/wrappers/terminal/gxm_to_gymnax.py
class GxmGymnaxWrapper(GymnaxWrapper):
    @property
    def default_params(self) -> None:
        return None

    def step(
        self,
        key: Key,
        state: Array,
        action: Array,
        params: Optional[EnvParams] = None,
    ) -> tuple[Array, Array, Array, Array, dict[str, Any]]:
        del params
        next_state, timestep = self._env.step(key, state, action)
        return (
            timestep.obs,
            next_state,
            timestep.reward,
            timestep.done,
            timestep.info,
        )

    def reset(
        self, key: Key, params: Optional[EnvParams] = None
    ) -> tuple[Array, Array]:
        del params
        state, timestep = self._env.init(key)
        return timestep.obs, state

    def action_space(self, params: Optional[EnvParams] = None) -> spaces.Space:
        del params
        return self._gxm_to_gymnax_space(self._env.action_space)

    def observation_space(self, params: Optional[EnvParams] = None) -> spaces.Space:
        del params
        return self._gxm_to_gymnax_space(self._env.observation_space)

    def _gxm_to_gymnax_space(self, space: Any) -> spaces.Space:
        from gxm.spaces import Box, Discrete, Tree

        if isinstance(space, Discrete):
            return spaces.Discrete(space.n)
        if isinstance(space, Box):
            return spaces.Box(space.low, space.high, space.shape, jnp.float32)
        if isinstance(space, Tree):
            if isinstance(space.spaces, (list, tuple)):
                return spaces.Tuple(
                    [self._gxm_to_gymnax_space(s) for s in space.spaces]
                )
            if isinstance(space.spaces, dict):
                return spaces.Dict(
                    {k: self._gxm_to_gymnax_space(v) for k, v in space.spaces.items()}
                )
        raise NotImplementedError(f"Gxm space type {type(space)} not supported.")


def make(env_id, **kwargs):
    import gxm

    env = gxm.make(env_id, **kwargs)
    env = GxmGymnaxWrapper(env)
    env_params = env.default_params
    return env, env_params
