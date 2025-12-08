from gymnax.environments import spaces
import numpy as np

from memorax.utils.wrappers import GymnaxWrapper


class NavixGymnaxWrapper(GymnaxWrapper):

    def reset(self, key, params=None):
        timestep = self._env.reset(key)
        return timestep.observation, timestep

    def step(self, key, state, action, params=None):
        timestep = self._env.step(state, action)
        return timestep.observation, timestep, timestep.reward, timestep.is_done(), {}

    def observation_space(self, params):
        return spaces.Box(
            low=self._env.observation_space.minimum,
            high=self._env.observation_space.maximum,
            shape=(np.prod(self._env.observation_space.shape),),
            dtype=self._env.observation_space.dtype,
        )

    def action_space(self, params):
        return spaces.Discrete(
            num_categories=self._env.action_space.maximum.item() + 1,
        )


def make(env_id, **kwargs):
    import navix as nx

    env = nx.make(env_id, **kwargs)
    env = NavixGymnaxWrapper(env)
    return env, None
