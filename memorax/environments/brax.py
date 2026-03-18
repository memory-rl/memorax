from typing import Any

import jax.numpy as jnp
from gymnax.environments import EnvParams, spaces

from memorax.environments.wrappers import GymnaxWrapper, MaskObservationWrapper
from memorax.utils.typing import Array, Key

masks = {
    "ant": {
        "F": jnp.ones(27, dtype=jnp.bool),
        "P": jnp.zeros(27, dtype=jnp.bool).at[:13].set(True),
        "V": jnp.zeros(27, dtype=jnp.bool).at[13:].set(True),
    },
    "halfcheetah": {
        "F": jnp.ones(17, dtype=jnp.bool),
        "P": jnp.zeros(17, dtype=jnp.bool).at[jnp.array([0, 1, 2, 3, 8, 9, 10, 11, 12])].set(True),
        "V": jnp.zeros(17, dtype=jnp.bool).at[jnp.array([4, 5, 6, 7, 13, 14, 15, 16])].set(True),
    },
    "hopper": {
        "F": jnp.ones(11, dtype=jnp.bool),
        "P": jnp.zeros(11, dtype=jnp.bool).at[:5].set(True),
        "V": jnp.zeros(11, dtype=jnp.bool).at[5:].set(True),
    },
    "walker2d": {
        "F": jnp.ones(17, dtype=jnp.bool),
        "P": jnp.zeros(17, dtype=jnp.bool).at[:8].set(True),
        "V": jnp.zeros(17, dtype=jnp.bool).at[8:].set(True),
    },
}


class BraxGymnaxWrapper(GymnaxWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.action_size = env.action_size
        self.observation_size = (env.observation_size,)

    @property
    def default_params(self) -> EnvParams:
        return EnvParams(max_steps_in_episode=1000)

    def reset(self, key: Key, params) -> tuple[Array, Any]:
        state = self._env.reset(key)
        return state.obs, state

    def step(self, key: Key, state, action: Array, params) -> tuple[Array, Any, Array, Array, dict]:
        next_state = self._env.step(state, action)
        return (
            next_state.obs,
            next_state,
            next_state.reward,
            next_state.done.astype(jnp.bool),
            {},
        )

    def observation_space(self, params) -> spaces.Box:
        return spaces.Box(
            low=-jnp.inf,
            high=jnp.inf,
            shape=(self._env.observation_size,),
        )

    def action_space(self, params) -> spaces.Box:
        return spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self._env.action_size,),
        )


def make(env_id: str, mode="F", backend="generalized", **kwargs) -> tuple:
    from brax import envs
    from brax.envs.wrappers.training import AutoResetWrapper, EpisodeWrapper

    env = envs.get_environment(env_name=env_id, backend=backend, **kwargs)
    env = EpisodeWrapper(env, episode_length=1000, action_repeat=1)
    env = AutoResetWrapper(env)
    env = BraxGymnaxWrapper(
        env,
    )
    env = MaskObservationWrapper(env, mask=masks[env_id][mode])

    env_params = env.default_params
    return env, env_params
