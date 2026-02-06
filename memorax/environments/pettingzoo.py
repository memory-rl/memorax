"""PettingZoo multi-agent environment wrapper for JAX-based training."""

from typing import Any, Callable, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np
from flax import struct
from gymnax.environments import spaces

from memorax.utils.typing import Array, Key

# Registry for named environments
_REGISTRY: dict[str, Callable[..., Tuple["PettingZooWrapper", Any]]] = {}


def register(name: str, factory: Callable[..., Tuple["PettingZooWrapper", Any]]) -> None:
    """Register an environment factory.

    Args:
        name: Environment name (e.g., "cogames").
        factory: Function that takes (env_id, num_envs, **kwargs) and returns (wrapper, env_info).
    """
    _REGISTRY[name] = factory


def make(
    env_id: Union[str, Callable[[], Any]], num_envs: int, **kwargs
) -> Tuple["PettingZooWrapper", Any]:
    """Create a PettingZoo environment wrapper.

    Args:
        env_id: Either a factory function that creates a PettingZoo ParallelEnv,
            or a string "namespace:env_name" for registered environments.
        num_envs: Number of parallel environments.
        **kwargs: Additional arguments passed to registered environment factories.

    Returns:
        Tuple of (PettingZooWrapper, env_info). env_info is None for direct
        factory functions, or environment-specific info for registered envs.

    Example (direct factory):
        ```python
        from pettingzoo.mpe import simple_spread_v3
        from memorax.environments.pettingzoo import make

        env, _ = make(simple_spread_v3.parallel_env, num_envs=8)
        ```

    Example (registered environment):
        ```python
        from memorax.environments import pettingzoo
        import memorax.environments.cogames  # registers "cogames"

        env, env_info = pettingzoo.make("cogames:cogsguard_arena.basic", num_envs=8)
        ```
    """
    if isinstance(env_id, str) and ":" in env_id:
        namespace, env_name = env_id.split(":", 1)
        if namespace not in _REGISTRY:
            raise ValueError(f"Unknown namespace: {namespace}. Available: {list(_REGISTRY.keys())}")
        return _REGISTRY[namespace](env_name, num_envs, **kwargs)

    # Direct factory function
    return PettingZooWrapper(env_id, num_envs), None


@struct.dataclass
class PettingZooEnvState:
    step: int = 0


class PettingZooWrapper:
    """Wraps vectorized PettingZoo ParallelEnv environments for multi-agent JAX training.

    This wrapper creates multiple instances of a PettingZoo environment and runs them
    in parallel, outputting stacked arrays compatible with MAPPO's vmap-based batching.

    The wrapper is designed to be vmapped by MAPPO - each call to reset/step operates
    on all internal environments but returns shapes that allow vmap to add the batch
    dimension correctly.

    Compatible with MAPPO and other multi-agent algorithms that expect:
    - observations: (num_agents, *obs_shape) per reset, vmapped to (num_agents, num_envs, *obs_shape)
    - actions: (num_agents,) per step, vmapped to (num_agents, num_envs)
    - rewards: (num_agents,) per step
    - dones: (num_agents,) per step

    Example:
        ```python
        from mettagrid.envs.pettingzoo_env import MettaGridPettingZooEnv
        from mettagrid.simulator import Simulator

        def make_env():
            simulator = Simulator()
            return MettaGridPettingZooEnv(simulator, cfg)

        env = PettingZooWrapper(make_env, num_envs=8)
        ```
    """

    def __init__(self, env_fn: Callable[[], Any], num_envs: int):
        """Initialize the wrapper.

        Args:
            env_fn: Factory function that creates a PettingZoo ParallelEnv instance.
            num_envs: Number of parallel environments to run.
        """
        self._env_fn = env_fn
        self._num_envs = num_envs

        # Create environments
        self._envs = [env_fn() for _ in range(num_envs)]

        # Get env info from first env
        first_env = self._envs[0]
        self._agent_ids = tuple(first_env.possible_agents)
        self._num_agents = len(self._agent_ids)

        # Get observation and action space info
        obs_space = first_env.observation_space(self._agent_ids[0])
        self._obs_shape = obs_space.shape
        self._obs_dtype = jnp.dtype(obs_space.dtype)

        action_space = first_env.action_space(self._agent_ids[0])
        self._num_actions = action_space.n

        # Build action_spaces dict for compatibility with MAPPO
        self._action_spaces = {
            aid: spaces.Discrete(self._num_actions) for aid in self._agent_ids
        }

        # Track current env index for vmap broadcasts
        self._env_idx = 0

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
        """Return True to indicate this is a pre-vectorized environment.

        MAPPO checks this property to determine whether to vmap over env calls.
        Pre-vectorized environments handle multiple envs internally, so MAPPO
        should call reset/step directly without vmap.
        """
        return True

    def reset(self, key: Key) -> Tuple[Array, PettingZooEnvState]:
        """Reset all environments.

        When vmapped by MAPPO, this is called num_envs times but broadcasts
        internally to reset all envs at once.

        Args:
            key: JAX random key (unused, but kept for API compatibility).

        Returns:
            Tuple of (observations, state) where observations has shape
            (num_agents, num_envs, *obs_shape) for compatibility with vmap out_axes=(1, 0).
        """

        def _reset(key):
            all_obs = []
            for env in self._envs:
                obs_dict, _ = env.reset()
                # Stack observations for this env: (num_agents, *obs_shape)
                env_obs = np.stack(
                    [obs_dict[aid] for aid in self._agent_ids], axis=0
                )
                all_obs.append(env_obs)

            # Stack across envs: (num_envs, num_agents, *obs_shape)
            obs = np.stack(all_obs, axis=0)
            # Transpose to (num_agents, num_envs, *obs_shape)
            obs = np.moveaxis(obs, 0, 1)
            return jnp.array(obs, dtype=self._obs_dtype)

        # Output shape: (num_agents, num_envs, *obs_shape)
        # When vmapped with out_axes=(1, 0), this gets broadcast correctly
        output_shape = (self._num_agents, self._num_envs, *self._obs_shape)

        obs = jax.pure_callback(
            _reset,
            jax.ShapeDtypeStruct(output_shape, self._obs_dtype),
            key,
            vmap_method="broadcast_all",
        )

        state = PettingZooEnvState(step=0)
        return obs, state

    def step(
        self,
        key: Key,
        state: PettingZooEnvState,
        actions: Array,
    ) -> Tuple[Array, PettingZooEnvState, Array, Array, dict]:
        """Step all environments.

        Args:
            key: JAX random key (unused, but kept for API compatibility).
            state: Current environment state.
            actions: Actions for all agents, shape (num_agents, num_envs).

        Returns:
            Tuple of (observations, new_state, rewards, dones, info) where:
            - observations: (num_agents, num_envs, *obs_shape)
            - rewards: (num_agents, num_envs)
            - dones: (num_agents, num_envs)
        """

        def _step(actions):
            # actions: (num_agents, num_envs)
            actions = np.asarray(actions, dtype=np.int32)

            all_obs = []
            all_rewards = []
            all_dones = []

            for env_idx, env in enumerate(self._envs):
                # Build action dict for this env
                action_dict = {
                    aid: actions[agent_idx, env_idx]
                    for agent_idx, aid in enumerate(self._agent_ids)
                }

                # Step the environment
                obs_dict, reward_dict, term_dict, trunc_dict, _ = env.step(action_dict)

                # Handle episode resets - if all agents are done, reset the env
                all_done = all(
                    term_dict.get(aid, False) or trunc_dict.get(aid, False)
                    for aid in self._agent_ids
                )
                if all_done:
                    obs_dict, _ = env.reset()

                # Stack per-agent data for this env
                env_obs = np.stack(
                    [obs_dict[aid] for aid in self._agent_ids], axis=0
                )
                env_rewards = np.array(
                    [reward_dict.get(aid, 0.0) for aid in self._agent_ids],
                    dtype=np.float32,
                )
                env_dones = np.array(
                    [
                        term_dict.get(aid, False) or trunc_dict.get(aid, False)
                        for aid in self._agent_ids
                    ],
                    dtype=np.bool_,
                )

                all_obs.append(env_obs)
                all_rewards.append(env_rewards)
                all_dones.append(env_dones)

            # Stack across envs: (num_envs, num_agents, ...)
            obs = np.stack(all_obs, axis=0)
            rewards = np.stack(all_rewards, axis=0)
            dones = np.stack(all_dones, axis=0)

            # Transpose to (num_agents, num_envs, ...)
            obs = np.moveaxis(obs, 0, 1)
            rewards = np.moveaxis(rewards, 0, 1)
            dones = np.moveaxis(dones, 0, 1)

            return (
                jnp.array(obs, dtype=self._obs_dtype),
                jnp.array(rewards, dtype=jnp.float32),
                jnp.array(dones, dtype=jnp.bool_),
            )

        # Output shapes
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

        new_state = PettingZooEnvState(step=state.step + 1)
        return obs, new_state, rewards, dones, {}
