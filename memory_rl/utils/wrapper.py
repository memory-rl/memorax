# Wrappers taken from https://github.com/luchris429/purejaxrl/blob/main/purejaxrl/wrappers.py

from functools import partial
from typing import Any, Optional, Tuple, Union

import chex
import jax
import jax.numpy as jnp
import navix as nx
import numpy as np
from brax import envs
from brax.envs.wrappers.training import AutoResetWrapper, EpisodeWrapper
from flax import struct
from gymnax.environments import environment, spaces, EnvParams


class GymnaxWrapper(object):
    """Base class for Gymnax wrappers."""

    def __init__(self, env):
        self._env = env

    # provide proxy access to regular attributes of wrapped object
    def __getattr__(self, name):
        return getattr(self._env, name)


@struct.dataclass
class WrappedParams:
    env_params: Any
    max_steps_in_episode: int


def _as_int(x) -> int:
    if isinstance(x, (int, np.integer)):
        return int(x)
    if hasattr(x, "item"):
        return int(x.item())
    return int(x)


class PopGymWrapper(GymnaxWrapper):
    def __init__(self, env):
        super().__init__(env)

    @property
    def default_params(self) -> WrappedParams:
        env_params = self._env.default_params
        max_steps = self._infer_max_steps(self._env, env_params)
        return WrappedParams(env_params=env_params, max_steps_in_episode=max_steps)

    def reset_env(
        self, key: chex.PRNGKey, params: WrappedParams
    ) -> Tuple[chex.Array, Any]:
        return self._env.reset_env(key, params.env_params)

    def step_env(
        self, key: chex.PRNGKey, state: Any, action: Any, params: WrappedParams
    ) -> Tuple[chex.Array, Any, float, bool, dict]:
        obs, new_state, reward, done, info = self._env.step_env(
            key, state, action, params.env_params
        )
        return obs, new_state, reward, done, info

    def action_space(self, params: Optional[WrappedParams] = None) -> spaces.Space:
        env_params = self._env_params(params)
        return self._env.action_space(env_params)

    def observation_space(self, params: Optional[WrappedParams] = None) -> spaces.Space:
        env_params = self._env_params(params)
        return self._env.observation_space(env_params)

    def state_space(self, params: Optional[WrappedParams] = None) -> spaces.Space:
        if hasattr(self._env, "state_space"):
            env_params = self._env_params(params)
            return self._env.state_space(env_params)
        return super().state_space(params)

    @property
    def name(self) -> str:
        return getattr(self._env, "name", self.env.__class__.__name__)

    @property
    def num_actions(self) -> int:
        return (
            getattr(self._env, "num_actions", None)
            or self._env.action_space(self.env.default_params).n
        )

    def _env_params(self, params: Optional[WrappedParams]):
        return self._env.default_params if (params is None) else params.env_params

    def _infer_max_steps(self, env, params) -> int:
        """
        Try common horizon fields first; fall back to simple derivations.
        Handles your Battleship, CartPole, Concentration, CountRecall,
        HigherLower, MineSweeper, MultiarmedBandit, Pendulum, Repeat*, and MetaCartPole.
        """
        # Direct fields on the env instance
        for attr in [
            "max_steps_in_episode",  # CartPole, Pendulum, etc.
            "max_episode_length",  # Battleship, MineSweeper
            "episode_length",  # MultiarmedBandit, Concentration (ceil expr)
        ]:
            if hasattr(env, attr):
                return _as_int(getattr(env, attr))

        # Derived by convention
        # Deck/card-based games
        if hasattr(env, "decksize") and hasattr(env, "num_decks"):
            return _as_int(getattr(env, "decksize") * getattr(env, "num_decks"))

        if hasattr(env, "num_cards"):
            return _as_int(getattr(env, "num_cards"))

        # Grid + mines (in case max_episode_length wasn't present)
        if hasattr(env, "dims") and hasattr(env, "num_mines"):
            dims = np.array(getattr(env, "dims"))
            return int(int(dims[0]) * int(dims[1]) - int(getattr(env, "num_mines")))

        # Meta envs (e.g., NoisyStatelessMetaCartPole)
        # If params has num_trials_per_episode and the env holds an inner env with a per-trial max,
        # report worst-case as trials * inner_max.
        if hasattr(params, "num_trials_per_episode") and hasattr(env, "env"):
            trials = _as_int(getattr(params, "num_trials_per_episode"))
            inner_env = getattr(env, "env")
            # Prefer inner explicit horizons; else recurse to derive from inner+its default params.
            inner_default = getattr(
                params, "env_params", getattr(inner_env, "default_params", None)
            )
            if callable(inner_default):
                inner_default = inner_default()
            inner_max = None
            for attr in [
                "max_steps_in_episode",
                "max_episode_length",
                "episode_length",
            ]:
                if hasattr(inner_env, attr):
                    inner_max = _as_int(getattr(inner_env, attr))
                    break
            if inner_max is None:
                # Recurse with inner env + its params
                inner_max = self._infer_max_steps(inner_env, inner_default)
            return int(trials * inner_max)

        raise ValueError(
            f"Could not infer max_steps_in_episode for {env.__class__.__name__}. "
            "Pass override_max_steps=... to PopGymWrapper."
        )


@struct.dataclass
class BSuiteEnvState:
    env_state: environment.EnvState
    episode_regret: float
    returned_episode_regret: float
    timestep: int


class BSuiteWrapper(GymnaxWrapper):
    """Expose per-step and cumulative regret via `info`."""

    def __init__(self, env: environment.Environment):
        super().__init__(env)

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: chex.PRNGKey, params: Optional[environment.EnvParams] = None):
        obs, env_state = self._env.reset(key, params)

        state = BSuiteEnvState(
            env_state=env_state,
            episode_regret=0.0,
            returned_episode_regret=0.0,
            timestep=0,
        )
        return obs, state

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        key: chex.PRNGKey,
        state: environment.EnvState,
        action: Union[int, float],
        params: Optional[environment.EnvParams] = None,
    ) -> Tuple[chex.Array, environment.EnvState, float, bool, dict]:
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


@struct.dataclass
class LogEnvState:
    env_state: environment.EnvState
    episode_returns: float
    episode_lengths: int
    returned_episode_returns: float
    returned_episode_lengths: int
    timestep: int


class LogWrapper(GymnaxWrapper):
    """Log the episode returns and lengths."""

    def __init__(self, env: environment.Environment):
        super().__init__(env)

    @partial(jax.jit, static_argnums=(0,))
    def reset(
        self, key: chex.PRNGKey, params: Optional[environment.EnvParams] = None
    ) -> Tuple[chex.Array, environment.EnvState]:
        obs, env_state = self._env.reset(key, params)
        state = LogEnvState(env_state, 0.0, 0, 0.0, 0, 0)
        return obs, state

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        key: chex.PRNGKey,
        state: environment.EnvState,
        action: Union[int, float],
        params: Optional[environment.EnvParams] = None,
    ) -> Tuple[chex.Array, environment.EnvState, float, bool, dict]:
        obs, env_state, reward, done, info = self._env.step(
            key, state.env_state, action, params
        )
        new_episode_return = state.episode_returns + reward
        new_episode_length = state.episode_lengths + 1
        state = LogEnvState(
            env_state=env_state,
            episode_returns=new_episode_return * (1 - done),
            episode_lengths=new_episode_length * (1 - done),
            returned_episode_returns=state.returned_episode_returns * (1 - done)
            + new_episode_return * done,
            returned_episode_lengths=state.returned_episode_lengths * (1 - done)
            + new_episode_length * done,
            timestep=state.timestep + 1,
        )
        info["returned_episode_returns"] = state.returned_episode_returns
        info["returned_episode_lengths"] = state.returned_episode_lengths
        info["timestep"] = state.timestep
        info["returned_episode"] = done
        return obs, state, reward, done, info


class BraxGymnaxWrapper:
    def __init__(self, env_name, backend="positional"):
        env = envs.get_environment(env_name=env_name, backend=backend)
        env = EpisodeWrapper(env, episode_length=1000, action_repeat=1)
        env = AutoResetWrapper(env)
        self._env = env
        self.action_size = env.action_size
        self.observation_size = (env.observation_size,)

    def reset(self, key, params=None):
        state = self._env.reset(key)
        return state.obs, state

    def step(self, key, state, action, params=None):
        next_state = self._env.step(state, action)
        return next_state.obs, next_state, next_state.reward, next_state.done > 0.5, {}

    def observation_space(self, params):
        return spaces.Box(
            low=-jnp.inf,
            high=jnp.inf,
            shape=(self._env.observation_size,),
        )

    def action_space(self, params):
        return spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self._env.action_size,),
        )


class NavixGymnaxWrapper:
    def __init__(self, env_name: str | None = None, env=None):
        if env is None:
            self._env = nx.make(env_name)
        else:
            self._env = env

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


class PixelCraftaxEnvWrapper(GymnaxWrapper):
    def __init__(self, env, normalize: bool = False):
        super().__init__(env)

        self.renderer = None

        self.normalize = normalize
        self.size = 110

    @partial(jax.jit, static_argnums=(0, -1))
    def reset(
        self, key: chex.PRNGKey, params: Optional[environment.EnvParams] = None
    ) -> Tuple[chex.Array, environment.EnvState]:
        image_obs, env_state = self._env.reset(key, params)
        # Craftax already returned normalized visual input
        image_obs = self.get_obs(image_obs, self.normalize)
        return image_obs, env_state

    @partial(jax.jit, static_argnums=(0, -1))
    def step(
        self,
        key: chex.PRNGKey,
        state: environment.EnvState,
        action: Union[int, float],
        params: Optional[environment.EnvParams] = None,
    ) -> Tuple[chex.Array, environment.EnvState, float, bool, dict]:
        image_obs, env_state, reward, done, info = self._env.step(
            key, state, action, params
        )
        image_obs = self.get_obs(image_obs, self.normalize)
        return image_obs, env_state, reward, done, info

    @partial(jax.jit, static_argnums=(0, 2))
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
