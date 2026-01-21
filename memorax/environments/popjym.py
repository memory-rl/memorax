from typing import Any

from flax import struct

from memorax.utils.wrappers import GymnaxWrapper

max_steps_in_episode = {
    "AutoencodeEasy": 105,
    "AutoencodeMedium": 209,
    "AutoencodeHard": 313,
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


@struct.dataclass(frozen=True)
class EnvParams:
    env_params: Any
    max_steps_in_episode: int


class PopJaxRLWrapper(GymnaxWrapper):
    def __init__(self, env):
        super().__init__(env)
        env_id = type(env).__name__
        self.max_steps_in_episode = max_steps_in_episode[env_id]

    def reset(self, key, params):
        return self._env.reset(key, params.env_params)

    def step(self, key, state, action, params):
        obs, new_state, reward, done, info = self._env.step(
            key, state, action, params.env_params
        )
        return obs, new_state, reward, done, info


def make(env_id, **kwargs):
    import popjym

    env, env_params = popjym.make(env_id, **kwargs)
    env = PopJaxRLWrapper(env)
    env_params = EnvParams(
        env_params=env_params, max_steps_in_episode=max_steps_in_episode[env_id]
    )
    return env, env_params
