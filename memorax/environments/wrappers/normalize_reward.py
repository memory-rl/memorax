from typing import Optional, Tuple, Union

import jax.numpy as jnp
from flax import struct
from gymnax.environments import environment

from memorax.utils.typing import Array, Key
from gymnax.wrappers.purerl import GymnaxWrapper


@struct.dataclass
class NormalizeRewardWrapperState:
    mean: float
    M2: float
    count: float
    G: float
    env_state: environment.EnvState


class NormalizeRewardWrapper(GymnaxWrapper):
    def __init__(self, env, gamma: float = 0.99, eps: float = 1e-8):
        super().__init__(env)
        self.gamma = gamma
        self.eps = eps

    def reset(
        self, key: Key, params: Optional[environment.EnvParams] = None
    ) -> Tuple[Array, NormalizeRewardWrapperState]:
        obs, env_state = self._env.reset(key, params)
        state = NormalizeRewardWrapperState(
            mean=0.0,
            M2=0.0,
            count=0.0,
            G=0.0,
            env_state=env_state,
        )
        return obs, state

    def step(
        self,
        key: Key,
        state: NormalizeRewardWrapperState,
        action: Union[int, float],
        params: Optional[environment.EnvParams] = None,
    ) -> Tuple[Array, NormalizeRewardWrapperState, float, bool, dict]:
        obs, env_state, reward, done, info = self._env.step(
            key, state.env_state, action, params
        )

        G = reward + self.gamma * state.G * (1 - done)

        count = state.count + 1
        delta = G - state.mean
        mean = state.mean + delta / count
        delta2 = G - mean
        M2 = state.M2 + delta * delta2
        var = M2 / count

        scaled_reward = reward / jnp.sqrt(var + self.eps)

        new_state = NormalizeRewardWrapperState(
            mean=mean,
            M2=M2,
            count=count,
            G=G * (1 - done),
            env_state=env_state,
        )
        return obs, new_state, scaled_reward, done, info
