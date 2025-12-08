import jax.numpy as jnp
import gymnax
from gymnax.environments import environment
from flax import struct

from memorax.utils.wrappers import GymnaxWrapper


@struct.dataclass
class BSuiteEnvState:
    env_state: environment.EnvState
    episode_regret: float
    returned_episode_regret: float
    timestep: int


class BSuiteWrapper(GymnaxWrapper):

    def __init__(self, env: environment.Environment):
        super().__init__(env)

    def reset(self, key, params):
        obs, env_state = self._env.reset(key, params)

        state = BSuiteEnvState(
            env_state=env_state,
            episode_regret=0.0,
            returned_episode_regret=0.0,
            timestep=0,
        )
        return obs, state

    def step(self, key, state, action, params):
        obs, env_state, reward, done, info = self._env.step(
            key, state.env_state, action, params
        )

        step_regret = jnp.where(jnp.logical_and(done, reward < 0), 2.0, 0.0)

        episode_regret = state.episode_regret + step_regret

        returned_episode_regret = jnp.where(done, episode_regret, 0.0)

        new_state = BSuiteEnvState(
            env_state=env_state,
            episode_regret=episode_regret * (1 - done),
            returned_episode_regret=returned_episode_regret,
            timestep=state.timestep + 1,
        )

        info["returned_episode_regret"] = returned_episode_regret
        info["returned_episode"] = done
        info["timestep"] = new_state.timestep

        info["step_regret"] = step_regret
        info["total_regret"] = episode_regret

        return obs, new_state, reward, done, info


def make(env_id, **kwargs):
    env, env_params = gymnax.make(env_id, **kwargs)

    if "bsuite" in env_id:
        env = BSuiteWrapper(env)

    return env, env_params
