import gxm
from gymnax.environments import spaces, EnvParams

from memory_rl.utils.wrappers import GymnaxWrapper


class GxmGymnaxWrapper(GymnaxWrapper):
    def __init__(self, env):
        super().__init__(env)

    def reset(self, key, params):
        state, timestep = self._env.init(key)
        return timestep.obs, state

    def step(self, key, state, action, params):
        next_state, timestep = self._env.step(key, state, action)
        return (
            timestep.obs,
            next_state,
            timestep.reward,
            timestep.done,
            timestep.info,
        )

    def action_space(self, params):
        return spaces.Discrete(num_categories=self._env.action_space.n)

    @property
    def default_params(self):
        return EnvParams(max_steps_in_episode=27_000)


def make(env_id, **kwargs):
    env = gxm.make(env_id, **kwargs)
    env = GxmGymnaxWrapper(env)
    env_params = env.default_params
    return env, env_params
