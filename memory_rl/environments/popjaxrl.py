from typing import Any
import jax
from gymnax.environments import EnvParams
from popjaxrl.envs import make as make_popjaxrl_env

from memory_rl.utils.wrappers import GymnaxWrapper

max_steps_in_episode = {
    "AutoencodeEasy": 104,
    "AutoencodeMedium": 208,
    "AutoencodeHard": 312,
    "BattleshipEasy": 64,
    "BattleshipMedium": 100,
    "BattleshipHard": 144,
    "StatelessCartPoleEasy": 200,
    "StatelessCartPoleMedium": 400,
    "StatelessCartPoleHard": 600,
    "NoisyStatelessCartPoleEasy": 200,
    "NoisyStatelessCartPoleMedium": 200,
    "NoisyStatelessCartPoleHard": 200,
    "ConcentrationEasy": 104,
    "ConcentrationMedium": 208,
    "ConcentrationHard": 104,
    "CountRecallEasy": 52,
    "CountRecallMedium": 104,
    "CountRecallHard": 208,
    "HigherLowerEasy": 52,
    "HigherLowerMedium": 104,
    "HigherLowerHard": 156,
    "RepeatFirstEasy": 52,
    "RepeatFirstMedium": 416,
    "RepeatFirstHard": 832,
    "RepeatPreviousEasy": 52,
    "RepeatPreviousMedium": 104,
    "RepeatPreviousHard": 156,
}


class EnvParams(EnvParams):
    env_params: Any


class PopGymWrapper(GymnaxWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.max_steps_in_episode = max_steps_in_episode[env.name]

    @property
    def default_params(self):
        env_params = self._env.default_params
        return EnvParams(
            env_params=env_params, max_steps_in_episode=self.max_steps_in_episode
        )

    def reset(self, key, params):
        return self._env.reset_env(key, params.env_params)

    def step(self, key, state, action, params):
        obs, new_state, reward, done, info = self._env.step_env(
            key, state, action, params.env_params
        )
        return obs, new_state, reward, done, info


def make(env_id: str, **kwargs):
    env, env_params = make_popjaxrl_env(env_id)
    env = PopGymWrapper(env)
    env_params = env.default_params
    return env, env_params
