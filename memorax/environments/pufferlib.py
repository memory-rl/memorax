"""PufferLib environment wrapper for JAX-based training."""

from typing import Any, Callable, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np
from flax import struct
from gymnax.environments import spaces

from memorax.utils.typing import Array, Key

# Registry for named environments
_REGISTRY: dict[str, Callable[..., Tuple["PufferLibWrapper", Any]]] = {}


def register(name: str, factory: Callable[..., Tuple["PufferLibWrapper", Any]]) -> None:
    """Register an environment factory.

    Args:
        name: Environment name (e.g., "cogames").
        factory: Function that takes (env_id, num_envs, **kwargs) and returns (wrapper, env_info).
    """
    _REGISTRY[name] = factory


def make(
    env_id: Union[str, Callable[[], Any]], num_envs: int, **kwargs
) -> Tuple["PufferLibWrapper", Any]:
    """Create a PufferLib environment wrapper.

    Args:
        env_id: Either a factory function that creates a single environment,
            or a string "namespace:env_name" for registered environments.
        num_envs: Number of parallel environments.
        **kwargs: Additional arguments passed to pufferlib.vector.make or
            registered environment factories.

    Returns:
        Tuple of (PufferLibWrapper, env_info). env_info is None for direct
        factory functions, or environment-specific info for registered envs.

    Example (direct factory):
        ```python
        from functools import partial
        from memorax.environments.pufferlib import make

        env, _ = make(
            partial(MyEnvClass, config=cfg),
            num_envs=8,
        )
        ```

    Example (registered environment):
        ```python
        from memorax.environments import pufferlib
        import memorax.environments.cogames  # registers "cogames"

        env, env_info = pufferlib.make("cogames:cogsguard_arena.basic", num_envs=8)
        ```
    """
    if isinstance(env_id, str) and ":" in env_id:
        namespace, env_name = env_id.split(":", 1)
        if namespace not in _REGISTRY:
            raise ValueError(f"Unknown namespace: {namespace}. Available: {list(_REGISTRY.keys())}")
        return _REGISTRY[namespace](env_name, num_envs, **kwargs)

    # Direct factory function
    import pufferlib.vector

    puffer_env = pufferlib.vector.make(env_id, num_envs=num_envs, **kwargs)
    return PufferLibWrapper(puffer_env), None


@struct.dataclass
class PufferLibEnvState:
    step: int = 0


class PufferLibWrapper:
    """Wraps a PufferLib vectorized environment for JAX-based training.

    This wrapper converts PufferLib's NumPy-based interface to JAX arrays
    using pure_callback for compatibility with JAX's functional paradigm.

    Example:
        ```python
        import pufferlib.vector

        puffer_env = pufferlib.vector.make(
            make_fn,
            num_envs=8,
        )
        env = PufferLibWrapper(puffer_env)
        ```
    """

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


class PufferLibMultiAgentWrapper:
    """Wraps a PufferLib vectorized environment for multi-agent JAX training.

    PufferLib flattens agents into the batch dimension: (num_envs * num_agents, ...).
    This wrapper reshapes to multi-agent format: (num_agents, num_envs, ...).

    This allows using PufferLib's parallel backends (Multiprocessing, Ray) with MAPPO.

    Example:
        ```python
        import pufferlib.vector

        puffer_env = pufferlib.vector.make(
            make_fn,
            num_envs=8,
            backend=pufferlib.vector.Multiprocessing,  # True parallelization
        )
        env = PufferLibMultiAgentWrapper(puffer_env, num_agents=4)
        ```
    """

    def __init__(self, env, num_agents: int):
        """Initialize the multi-agent wrapper.

        Args:
            env: PufferLib vectorized environment.
            num_agents: Number of agents per environment.
        """
        self._env = env
        self._num_agents = num_agents
        # PufferLib's num_envs is actually num_envs * num_agents
        self._num_envs = env.num_envs // num_agents

        obs_space = env.single_observation_space
        self._obs_shape = obs_space.shape
        self._obs_dtype = jnp.dtype(obs_space.dtype)

        self._num_actions = env.single_action_space.n

        # Build action_spaces dict for MAPPO compatibility
        self._agent_ids = tuple(range(num_agents))
        self._action_spaces = {
            aid: spaces.Discrete(self._num_actions) for aid in self._agent_ids
        }

    @property
    def num_envs(self) -> int:
        return self._num_envs

    @property
    def agents(self) -> tuple:
        return self._agent_ids

    @property
    def num_agents(self) -> int:
        return self._num_agents

    @property
    def action_spaces(self) -> dict:
        return self._action_spaces

    @property
    def obs_shape(self) -> tuple:
        return self._obs_shape

    @property
    def obs_dtype(self):
        return self._obs_dtype

    @property
    def num_actions(self) -> int:
        return self._num_actions

    @property
    def vectorized(self) -> bool:
        """This is a pre-vectorized environment."""
        return True

    def reset(self, key: Key) -> Tuple[Array, PufferLibEnvState]:
        """Reset all environments.

        Returns:
            Tuple of (observations, state) where observations has shape
            (num_agents, num_envs, *obs_shape).
        """

        def _reset(key):
            obs, _ = self._env.reset()
            # obs: (num_envs * num_agents, *obs_shape)
            # Reshape to (num_envs, num_agents, *obs_shape)
            obs = obs.reshape(self._num_envs, self._num_agents, *self._obs_shape)
            # Transpose to (num_agents, num_envs, *obs_shape)
            obs = np.moveaxis(obs, 1, 0)
            return jnp.array(obs, dtype=self._obs_dtype)

        output_shape = (self._num_agents, self._num_envs, *self._obs_shape)

        obs = jax.pure_callback(
            _reset,
            jax.ShapeDtypeStruct(output_shape, self._obs_dtype),
            key,
            vmap_method="broadcast_all",
        )

        state = PufferLibEnvState(step=0)
        return obs, state

    def step(
        self,
        key: Key,
        state: PufferLibEnvState,
        actions: Array,
    ) -> Tuple[Array, PufferLibEnvState, Array, Array, dict]:
        """Step all environments.

        Args:
            key: JAX random key (unused).
            state: Current environment state.
            actions: Actions for all agents, shape (num_agents, num_envs).

        Returns:
            Tuple of (observations, new_state, rewards, dones, info).
        """

        def _step(actions):
            # actions: (num_agents, num_envs)
            actions = np.asarray(actions, dtype=np.int32)
            # Transpose to (num_envs, num_agents) then flatten
            actions = np.moveaxis(actions, 0, 1).flatten()

            obs, rewards, dones, truncs, infos = self._env.step(actions)

            # Reshape outputs: (num_envs * num_agents,) -> (num_agents, num_envs)
            obs = obs.reshape(self._num_envs, self._num_agents, *self._obs_shape)
            obs = np.moveaxis(obs, 1, 0)

            rewards = rewards.reshape(self._num_envs, self._num_agents)
            rewards = np.moveaxis(rewards, 1, 0)

            combined_dones = dones | truncs
            combined_dones = combined_dones.reshape(self._num_envs, self._num_agents)
            combined_dones = np.moveaxis(combined_dones, 1, 0)

            return (
                jnp.array(obs, dtype=self._obs_dtype),
                jnp.array(rewards, dtype=jnp.float32),
                jnp.array(combined_dones, dtype=jnp.bool_),
            )

        obs_shape = (self._num_agents, self._num_envs, *self._obs_shape)
        scalar_shape = (self._num_agents, self._num_envs)

        obs, rewards, dones = jax.pure_callback(
            _step,
            (
                jax.ShapeDtypeStruct(obs_shape, self._obs_dtype),
                jax.ShapeDtypeStruct(scalar_shape, jnp.float32),
                jax.ShapeDtypeStruct(scalar_shape, jnp.bool_),
            ),
            actions,
            vmap_method="broadcast_all",
        )

        new_state = PufferLibEnvState(step=state.step + 1)
        return obs, new_state, rewards, dones, {}
