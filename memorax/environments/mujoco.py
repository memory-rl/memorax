import jax
import jax.numpy as jnp
from gymnax.environments import spaces

from memorax.utils.wrappers import GymnaxWrapper


class TimeLimitWrapper(GymnaxWrapper):
    def __init__(self, env, episode_length):
        super().__init__(env)
        self.episode_length = episode_length

    def reset(self, rng: jax.Array):
        state = self._env.reset(rng)
        state.info["steps"] = jnp.zeros((rng.shape[:-1]), dtype=jnp.int32)
        return state

    def step(self, state, action: jax.Array):
        state = self._env.step(state, action)
        steps = state.info["steps"] + 1
        limit = jnp.array(self.episode_length, dtype=jnp.int32)
        done = jnp.where(steps >= limit, jnp.ones_like(state.done), state.done)

        state.info["steps"] = steps
        return state.replace(done=done)


class AutoResetWrapper(GymnaxWrapper):
    def reset(self, rng: jax.Array):
        state = self._env.reset(rng)
        state.info["first_data"] = state.data
        state.info["first_obs"] = state.obs
        return state

    def step(self, state, action: jax.Array):
        if "steps" in state.info:
            steps = state.info["steps"]
            steps = jnp.where(state.done, jnp.zeros_like(steps), steps)
            state.info.update(steps=steps)
        state = state.replace(done=jnp.zeros_like(state.done))
        state = self._env.step(state, action)

        def where_done(x, y):
            done = state.done
            if done.shape and done.shape[0] != x.shape[0]:
                return y
            if done.shape:
                done = jnp.reshape(done, [x.shape[0]] + [1] * (len(x.shape) - 1))  # type: ignore
            return jnp.where(done, x, y)

        data = jax.tree.map(where_done, state.info["first_data"], state.data)
        obs = jax.tree.map(where_done, state.info["first_obs"], state.obs)
        return state.replace(data=data, obs=obs)


class MuJoCoGymnaxWrapper(GymnaxWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.timestep = 0

    def reset(self, key, params):
        state = self._env.reset(key)
        return state.obs, state

    def step(self, key, state, action, params):
        next_state = self._env.step(state, action)
        return (next_state.obs, next_state, next_state.reward, next_state.done, {})

    def action_space(self, params):
        n = self._env.action_size
        return spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(n,),
        )


def make(env_id, **kwargs):
    from ml_collections import FrozenConfigDict
    from mujoco_playground import registry

    env = registry.load(env_id, **kwargs)
    env_params = registry.get_default_config(env_id)
    env_params["max_steps_in_episode"] = env_params.episode_length + 1
    env = TimeLimitWrapper(env, env_params.episode_length)
    env = AutoResetWrapper(env)
    env = MuJoCoGymnaxWrapper(env)
    return env, FrozenConfigDict(env_params)
