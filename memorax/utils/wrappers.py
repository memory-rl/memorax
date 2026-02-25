from functools import partial
from typing import Callable, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np
from flax import struct
from gymnax.environments import environment, spaces

from memorax.utils.typing import Array, Key


class GymnaxWrapper:
    """Base class for Gymnax wrappers."""

    def __init__(self, env):
        self._env = env

    def __getattr__(self, name):
        return getattr(self._env, name)


class MaskObservationWrapper(GymnaxWrapper):
    def __init__(self, env, mask_dims: list, **kwargs):
        super().__init__(env)
        self.mask_dims = jnp.array(mask_dims, dtype=int)

    def observation_space(self, params) -> spaces.Box:
        assert isinstance(
            self._env.observation_space(params), spaces.Box
        ), "Only Box spaces are supported for now."
        low = self._env.observation_space(params).low
        if isinstance(low, jnp.ndarray):
            low = low[self.mask_dims]

        high = self._env.observation_space(params).high
        if isinstance(high, jnp.ndarray):
            high = high[self.mask_dims]

        return spaces.Box(
            low=low,
            high=high,
            shape=(self.mask_dims.shape[0],),
            dtype=self._env.observation_space(params).dtype,
        )

    @partial(jax.jit, static_argnums=(0, -1))
    def reset(
        self, key: Key, params: Optional[environment.EnvParams] = None
    ) -> Tuple[Array, environment.EnvState]:
        obs, state = self._env.reset(key, params)
        obs = obs[self.mask_dims]
        return obs, state

    @partial(jax.jit, static_argnums=(0, -1))
    def step(
        self,
        key: Key,
        state: environment.EnvState,
        action: Union[int, float],
        params: Optional[environment.EnvParams] = None,
    ) -> Tuple[Array, environment.EnvState, float, bool, dict]:
        obs, state, reward, done, info = self._env.step(key, state, action, params)
        obs = obs[self.mask_dims]
        return obs, state, reward, done, info


@struct.dataclass
class PeriodicObservationWrapperState:
    step: int
    env_state: environment.EnvState


class PeriodicObservationWrapper(GymnaxWrapper):
    def __init__(self, env, period: int, fill_fn: Callable[[Key, tuple], Array] = lambda key, shape: jnp.zeros(shape)):
        super().__init__(env)
        self.period = period
        self.fill_fn = fill_fn

    @partial(jax.jit, static_argnums=(0, -1))
    def reset(
        self, key: Key, params: Optional[environment.EnvParams] = None
    ) -> Tuple[Array, environment.EnvState]:
        obs, state = self._env.reset(key, params)
        state = PeriodicObservationWrapperState(step=0, env_state=state)
        return obs, state

    @partial(jax.jit, static_argnums=(0, -1))
    def step(
        self,
        key: Key,
        state: PeriodicObservationWrapperState,
        action: Union[int, float],
        params: Optional[environment.EnvParams] = None,
    ) -> Tuple[Array, environment.EnvState, float, bool, dict]:
        key, fill_key = jax.random.split(key)
        obs, env_state, reward, done, info = self._env.step(
            key, state.env_state, action, params
        )
        new_step = state.step + 1
        visible = new_step % self.period == 0
        fill = self.fill_fn(fill_key, obs.shape)
        obs = jnp.where(visible, obs, fill)
        state = PeriodicObservationWrapperState(step=new_step, env_state=env_state)
        return obs, state, reward, done, info


class FlickeringObservationWrapper(GymnaxWrapper):
    def __init__(
        self, env, p: float, fill_fn: Callable[[Key, tuple], Array] = lambda key, shape: jnp.zeros(shape)
    ):
        super().__init__(env)
        self.p = p
        self.fill_fn = fill_fn

    @partial(jax.jit, static_argnums=(0, -1))
    def step(
        self,
        key: Key,
        state: environment.EnvState,
        action: Union[int, float],
        params: Optional[environment.EnvParams] = None,
    ) -> Tuple[Array, environment.EnvState, float, bool, dict]:
        key, flicker_key, fill_key = jax.random.split(key, 3)
        obs, state, reward, done, info = self._env.step(key, state, action, params)
        visible = jax.random.uniform(flicker_key) >= self.p
        fill = self.fill_fn(fill_key, obs.shape)
        obs = jnp.where(visible, obs, fill)
        return obs, state, reward, done, info


@struct.dataclass
class DelayedObservationWrapperState:
    buffer: jnp.ndarray
    env_state: environment.EnvState


class DelayedObservationWrapper(GymnaxWrapper):
    def __init__(self, env, delay: int):
        super().__init__(env)
        self.delay = delay

    @partial(jax.jit, static_argnums=(0, -1))
    def reset(
        self, key: Key, params: Optional[environment.EnvParams] = None
    ) -> Tuple[Array, environment.EnvState]:
        obs, env_state = self._env.reset(key, params)
        buffer = jnp.zeros((self.delay,) + obs.shape)
        buffer = buffer.at[0].set(obs)
        state = DelayedObservationWrapperState(buffer=buffer, env_state=env_state)
        return jnp.zeros_like(obs), state

    @partial(jax.jit, static_argnums=(0, -1))
    def step(
        self,
        key: Key,
        state: DelayedObservationWrapperState,
        action: Union[int, float],
        params: Optional[environment.EnvParams] = None,
    ) -> Tuple[Array, environment.EnvState, float, bool, dict]:
        obs, env_state, reward, done, info = self._env.step(
            key, state.env_state, action, params
        )
        delayed_obs = state.buffer[-1]
        buffer = jnp.roll(state.buffer, shift=1, axis=0)
        buffer = buffer.at[0].set(obs)
        state = DelayedObservationWrapperState(buffer=buffer, env_state=env_state)
        return delayed_obs, state, reward, done, info


@struct.dataclass
class NormalizeObservationWrapperState:
    mean: jnp.ndarray
    var: jnp.ndarray
    count: float
    env_state: environment.EnvState


class NormalizeObservationWrapper(GymnaxWrapper):
    def __init__(self, env):
        super().__init__(env)

    def reset(self, key, params=None):
        obs, state = self._env.reset(key, params)
        state = NormalizeObservationWrapperState(
            mean=jnp.zeros_like(obs),
            var=jnp.ones_like(obs),
            count=1e-4,
            env_state=state,
        )
        batch_mean = jnp.mean(obs, axis=0)
        batch_var = jnp.var(obs, axis=0)
        batch_count = obs.shape[0]

        delta = batch_mean - state.mean
        tot_count = state.count + batch_count

        new_mean = state.mean + delta * batch_count / tot_count
        m_a = state.var * state.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + jnp.square(delta) * state.count * batch_count / tot_count
        new_var = M2 / tot_count
        new_count = tot_count

        state = NormalizeObservationWrapperState(
            mean=new_mean,
            var=new_var,
            count=new_count,
            env_state=state.env_state,
        )

        return (obs - state.mean) / jnp.sqrt(state.var + 1e-8), state

    def step(self, key, state, action, params=None):
        obs, env_state, reward, done, info = self._env.step(
            key, state.env_state, action, params
        )

        batch_mean = jnp.mean(obs, axis=0)
        batch_var = jnp.var(obs, axis=0)
        batch_count = obs.shape[0]

        delta = batch_mean - state.mean
        tot_count = state.count + batch_count

        new_mean = state.mean + delta * batch_count / tot_count
        m_a = state.var * state.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + jnp.square(delta) * state.count * batch_count / tot_count
        new_var = M2 / tot_count
        new_count = tot_count

        state = NormalizeObservationWrapperState(
            mean=new_mean,
            var=new_var,
            count=new_count,
            env_state=env_state,
        )
        return (
            (obs - state.mean) / jnp.sqrt(state.var + 1e-8),
            state,
            reward,
            done,
            info,
        )


class NoisyObservationWrapper(GymnaxWrapper):
    def __init__(
        self,
        env,
        noise_dims: list,
        noise_fn: Callable[[Key, tuple], Array],
    ):
        super().__init__(env)
        self.noise_dims = jnp.array(noise_dims, dtype=int)
        self.noise_fn = noise_fn

    def _add_noise(self, key: Key, obs: Array) -> Array:
        noise = jnp.zeros_like(obs)
        sampled = self.noise_fn(key, self.noise_dims.shape)
        noise = noise.at[self.noise_dims].set(sampled)
        return obs + noise

    @partial(jax.jit, static_argnums=(0, -1))
    def reset(
        self, key: Key, params: Optional[environment.EnvParams] = None
    ) -> Tuple[Array, environment.EnvState]:
        key, noise_key = jax.random.split(key)
        obs, state = self._env.reset(key, params)
        return self._add_noise(noise_key, obs), state

    @partial(jax.jit, static_argnums=(0, -1))
    def step(
        self,
        key: Key,
        state: environment.EnvState,
        action: Union[int, float],
        params: Optional[environment.EnvParams] = None,
    ) -> Tuple[Array, environment.EnvState, float, bool, dict]:
        key, noise_key = jax.random.split(key)
        obs, state, reward, done, info = self._env.step(key, state, action, params)
        return self._add_noise(noise_key, obs), state, reward, done, info


class ClipActionWrapper(GymnaxWrapper):
    def __init__(self, env, low=-1.0, high=1.0):
        super().__init__(env)
        self.low = low
        self.high = high

    def step(self, key, state, action, params=None):
        action = jnp.clip(action, self.low, self.high)
        return self._env.step(key, state, action, params)


class ScaleRewardWrapper(GymnaxWrapper):
    def __init__(self, env, scale=1.0):
        super().__init__(env)
        self.scale = scale

    def step(self, key, state, action, params=None):
        obs, env_state, reward, done, info = self._env.step(key, state, action, params)
        return obs, env_state, self.scale * reward, done, info


@struct.dataclass
class PufferLibEnvState:
    step: int = 0


class PufferLibWrapper:

    def __init__(self, env):
        self._env = env
        self.num_envs = env.num_envs

        obs_space = env.single_observation_space
        self.obs_shape = obs_space.shape
        self.obs_dtype = jnp.dtype(obs_space.dtype)

        self.num_actions = env.single_action_space.n

    @property
    def default_params(self):
        return None

    def reset(self, key: Key, params=None) -> Tuple[Array, PufferLibEnvState]:

        def _reset(key):
            obs, _ = self._env.reset()
            return jnp.array(obs, dtype=self.obs_dtype)

        obs = jax.pure_callback(
            _reset,
            jax.ShapeDtypeStruct(self.obs_shape, self.obs_dtype),
            key,
            vmap_method="broadcast_all",
        )

        state = PufferLibEnvState(step=0)
        return obs, state

    def step(
        self,
        key: Key,
        state: PufferLibEnvState,
        action: Array,
        params=None,
    ) -> Tuple[Array, PufferLibEnvState, Array, Array, dict]:

        def _step(action):
            action = np.asarray(action, dtype=np.int32)
            obs, rewards, dones, truncs, infos = self._env.step(action)

            return (
                jnp.array(obs, dtype=self.obs_dtype),
                jnp.array(rewards, dtype=jnp.float32),
                jnp.array(dones | truncs, dtype=jnp.bool_),
            )

        obs, rewards, dones = jax.pure_callback(
            _step,
            (
                jax.ShapeDtypeStruct(self.obs_shape, self.obs_dtype),
                jax.ShapeDtypeStruct((), jnp.float32),
                jax.ShapeDtypeStruct((), jnp.bool_),
            ),
            action,
            vmap_method="broadcast_all",
        )

        new_state = PufferLibEnvState(step=state.step + 1)
        return obs, new_state, rewards, dones, {}

    def observation_space(self, params=None) -> spaces.Box:
        """Return the observation space."""
        return spaces.Box(
            low=-jnp.inf,
            high=jnp.inf,
            shape=self.obs_shape,
            dtype=self.obs_dtype,
        )

    def action_space(self, params=None) -> spaces.Discrete:
        """Return the action space."""
        return spaces.Discrete(self.num_actions)


class FlattenMultiAgentObservationWrapper:
    """Wrapper that flattens multi-dimensional observations.

    Useful for environments with observations like (height, width, channels)
    that need to be flattened for MLP-based policies.
    """

    def __init__(self, env):
        self._env = env
        self.num_envs = env.num_envs
        self.num_actions = env.num_actions

        # Flatten observation shape
        self._raw_obs_shape = env.obs_shape
        self.obs_shape = (int(np.prod(env.obs_shape)),)
        self.obs_dtype = env.obs_dtype

    @property
    def default_params(self):
        return self._env.default_params

    def _flatten_obs(self, obs: Array) -> Array:
        """Flatten observations: (..., *obs_dims) -> (..., flat_dim)."""
        return obs.reshape(obs.shape[: -len(self._raw_obs_shape)] + (-1,))

    def reset(self, key: Key, params=None) -> Tuple[Array, PufferLibEnvState]:
        obs, state = self._env.reset(key, params)
        return self._flatten_obs(obs), state

    def step(
        self,
        key: Key,
        state: PufferLibEnvState,
        action: Array,
        params=None,
    ) -> Tuple[Array, PufferLibEnvState, Array, Array, dict]:
        obs, state, rewards, dones, info = self._env.step(key, state, action, params)
        return self._flatten_obs(obs), state, rewards, dones, info

    def observation_space(self, params=None) -> spaces.Box:
        """Return the flattened observation space."""
        return spaces.Box(
            low=-jnp.inf,
            high=jnp.inf,
            shape=self.obs_shape,
            dtype=self.obs_dtype,
        )

    def action_space(self, params=None) -> spaces.Discrete:
        """Return the action space."""
        return spaces.Discrete(self.num_actions)
