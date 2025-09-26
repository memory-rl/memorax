import jax
import gxm
from gymnax.environments import spaces, EnvParams

from memory_rl.utils.wrappers import GymnaxWrapper


class GxmGymnaxWrapper(GymnaxWrapper):
    def __init__(self, env, key):
        super().__init__(env)

    def reset(self, key, params):
        state = self._env.init(key)
        return state.obs, state

    def step(self, key, state, action, params):
        next_state = self._env.step(key, state, action)
        return (
            next_state.obs,
            next_state,
            next_state.reward,
            next_state.done,
            next_state.info,
        )

    def action_space(self, params):
        return spaces.Discrete(num_categories=self._env.n)

    @property
    def default_params(self):
        return EnvParams(max_steps_in_episode=1000)


def make(env_id: str, key: jax.Array, **kwargs):
    env = gxm.make(env_id)
    env = GxmGymnaxWrapper(env, key)
    env_params = env.default_params
    return env, env_params
