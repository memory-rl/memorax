from functools import partial
from typing import Any
from dataclasses import dataclass

from gymnax.environments import spaces
import jax
import jax.numpy as jnp
from popjaxrl.envs import make as make_popjaxrl_env

from memory_rl.utils.wrappers import GymnaxWrapper

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


class AliasPrevActionV2(GymnaxWrapper):

    def observation_space(self, params):
        action_space = self._env.action_space(params)
        if type(action_space) == spaces.Discrete:
            low = jnp.concatenate([self._env.observation_space(params).low, jnp.zeros((action_space.n+1,))])
            high = jnp.concatenate([self._env.observation_space(params).high, jnp.ones((action_space.n+1,))])
            return spaces.Box(
                low=low,
                high=high,
                shape=(self._env.observation_space(params).shape[-1]+action_space.n+1,), 
                dtype=self._env.observation_space(params).dtype,
            )
        elif type(action_space) == spaces.Box:
            low = jnp.concatenate([self._env.observation_space(params).low, jnp.array([action_space.low]), jnp.array([0.0])])
            high = jnp.concatenate([self._env.observation_space(params).high, jnp.array([action_space.high]), jnp.array([1.0])])
            return spaces.Box(
                low=low,
                high=high,
                shape=(self._env.observation_space(params).shape[-1]+2,),
                dtype=self._env.observation_space(params).dtype,
            )
        else:
            raise NotImplementedError

    @partial(jax.jit, static_argnums=(0,))
    def reset(
        self, key, params = None
    ):
        action_space = self._env.action_space(params)
        obs, state = self._env.reset(key, params)
        if isinstance(action_space, spaces.Box):
            obs = jnp.concatenate([obs, jnp.array([0.0, 1.0])])
        elif isinstance(action_space, spaces.Discrete):
            obs = jnp.concatenate([obs, jnp.zeros((action_space.n,)), jnp.array([1.0])])
        else:
            raise NotImplementedError
        return obs, state

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        key,
        state,
        action,
        params = None,
    ):
        obs, state, reward, done, info = self._env.step(
            key, state, action, params
        )
        action_space = self._env.action_space(params)
        if isinstance(action_space, spaces.Discrete):
            action_in = jnp.zeros((action_space.n,))
            action_in = action_in.at[action].set(1.0)
            obs = jnp.concatenate([obs, action_in, jnp.array([0.0])])
        else:
            obs = jnp.concatenate([obs, action, jnp.array([0.0])])
        return obs, state, reward, done, info

@dataclass(frozen=True)
class EnvParams:
    env_params: Any
    max_steps_in_episode: int

class PopJaxRLWrapper(GymnaxWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.max_steps_in_episode = max_steps_in_episode[env.name]

    def reset(self, key, params):
        return self._env.reset(key, params.env_params)

    def step(self, key, state, action, params):
        obs, new_state, reward, done, info = self._env.step(
            key, state, action, params.env_params
        )
        return obs, new_state, reward, done, info


def make(env_id: str, **kwargs):
    env, env_params = make_popjaxrl_env(env_id)
    env = AliasPrevActionV2(env)
    env = PopJaxRLWrapper(env)
    env_params = EnvParams(
        env_params=env_params, max_steps_in_episode=max_steps_in_episode[env_id]
    )
    return env, env_params
