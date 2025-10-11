from typing import Any
from flax.struct import dataclass
import jax
from craftax import craftax_env
from gymnax.environments import spaces

from memory_rl.utils.wrappers import GymnaxWrapper


class PixelCraftaxEnvWrapper(GymnaxWrapper):
    def __init__(self, env, normalize: bool = False):
        super().__init__(env)

        self.renderer = None

        self.normalize = normalize
        self.size = 110

    def reset(self, key, params):
        image_obs, env_state = self._env.reset(key, params)
        image_obs = self.get_obs(image_obs, self.normalize)
        return image_obs, env_state

    def step(self, key, state, action, params):
        image_obs, env_state, reward, done, info = self._env.step(
            key, state, action, params
        )
        image_obs = self.get_obs(image_obs, self.normalize)
        return image_obs, env_state, reward, done, info

    def get_obs(self, obs, normalize):
        if not normalize:
            obs *= 255
        assert len(obs.shape) == 4
        obs = obs[:27, :, :]
        return obs

    def observation_space(self, params):
        low, high = 0, 255
        if self.normalize:
            high = 1
        return spaces.Box(
            low=low,
            high=high,
            shape=(
                27,
                33,
                3,
            ),
        )


class CraftaxWrapper(GymnaxWrapper):

    def reset(self, key, params):
        return self._env.reset(key, params.env_params)

    def step(self, key, state, action, params):
        obs, new_state, reward, done, info = self._env.step(
            key, state, action, params.env_params
        )
        return obs, new_state, reward, done, info


@dataclass(frozen=True)
class EnvParams:
    env_params: Any
    max_steps_in_episode: int


def make(env_id, **kwargs):
    env = craftax_env.make_craftax_env_from_name(env_id, **kwargs)

    if env_id == "Craftax-Pixels-v1":
        env = PixelCraftaxEnvWrapper(env)

    env = CraftaxWrapper(env)

    env_params = env.default_params
    env_params = EnvParams(
        env_params=env_params,
        max_steps_in_episode=env_params.max_timesteps,
    )

    return env, env_params
