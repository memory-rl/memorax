import jax
import jax.numpy as jnp
from brax import envs
from brax.envs.wrappers.training import AutoResetWrapper, EpisodeWrapper
from gymnax.environments import spaces, EnvParams

from memory_rl.utils.wrappers import GymnaxWrapper, MaskObservationWrapper

mask_dims = {
    "ant": {
        "F": list(range(27)),
        "P": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        "V": [13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26],
    },
    "half_cheetah": {
        "F": list(range(17)),
        "P": [0, 1, 2, 3, 8, 9, 10, 11, 12],
        "V": [4, 5, 6, 7, 13, 14, 15, 16],
    },
    "hopper": {
        "F": list(range(11)),
        "P": [0, 1, 2, 3, 4],
        "V": [5, 6, 7, 8, 9, 10],
    },
    "walker": {
        "F": list(range(17)),
        "P": [0, 1, 2, 3, 4, 5, 6, 7],
        "V": [8, 9, 10, 11, 12, 13, 14, 15, 16],
    },
}


class BraxGymnaxWrapper(GymnaxWrapper):
    def __init__(self, env_name, backend="mjx"):
        env = envs.get_environment(env_name=env_name, backend=backend)
        env = EpisodeWrapper(env, episode_length=1000, action_repeat=1)
        env = AutoResetWrapper(env)
        super().__init__(env)
        self.action_size = env.action_size
        self.observation_size = (env.observation_size,)

    @property
    def default_params(self) -> EnvParams:
        return EnvParams(max_steps_in_episode=1000)

    def reset(self, key, params):
        state = self._env.reset(key)
        return state.obs, state

    def step(self, key, state, action, params):
        next_state = self._env.step(state, action)
        return next_state.obs, next_state, next_state.reward, next_state.done, {}

    def observation_space(self, params):
        return spaces.Box(
            low=-jnp.inf,
            high=jnp.inf,
            shape=(self._env.observation_size,),
        )

    def action_space(self, params):
        return spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self._env.action_size,),
        )


def make(env_id: str, **kwargs):
    env_id, mask_mode = env_id.split("-")
    env = BraxGymnaxWrapper(env_id, backend="mjx")

    env = MaskObservationWrapper(env, mask_dims=mask_dims[env_id][mask_mode])

    env_params = env.default_params
    return env, env_params
