"""Direct MettaGrid environment wrapper for JAX-based multi-agent training.

This wrapper interfaces directly with MettaGrid's Simulation class, bypassing
PufferLib and PettingZoo for better control over vectorization and multi-agent output format.
"""

from typing import Any, Callable, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np
from flax import struct
from gymnax.environments import spaces

from memorax.utils.typing import Array, Key

# Registry for named environments
_REGISTRY: dict[str, Callable[..., Tuple["MettaGridWrapper", Any]]] = {}


def register(name: str, factory: Callable[..., Tuple["MettaGridWrapper", Any]]) -> None:
    """Register an environment factory."""
    _REGISTRY[name] = factory


def make(
    env_id: Union[str, Any], num_envs: int, **kwargs
) -> Tuple["MettaGridWrapper", Any]:
    """Create a MettaGrid environment wrapper.

    Args:
        env_id: Either a MettaGridConfig object, or a string "namespace:env_name"
            for registered environments (e.g., "cogames:cogsguard_arena.basic").
        num_envs: Number of parallel environments.
        **kwargs: Additional arguments passed to the factory.

    Returns:
        Tuple of (MettaGridWrapper, env_info).

    Example:
        ```python
        from memorax.environments import mettagrid
        import memorax.environments.cogames  # registers "cogames"

        env, env_info = mettagrid.make("cogames:cogsguard_arena.basic", num_envs=8)
        ```
    """
    if isinstance(env_id, str) and ":" in env_id:
        namespace, env_name = env_id.split(":", 1)
        if namespace not in _REGISTRY:
            raise ValueError(f"Unknown namespace: {namespace}. Available: {list(_REGISTRY.keys())}")
        return _REGISTRY[namespace](env_name, num_envs, **kwargs)

    # Direct config object
    return MettaGridWrapper(config=env_id, num_envs=num_envs, **kwargs), None


@struct.dataclass
class MettaGridEnvState:
    step: int = 0


class MettaGridWrapper:
    """Wraps MettaGrid Simulation directly for multi-agent JAX training.

    Creates multiple Simulation instances and outputs stacked arrays in
    multi-agent format: (num_agents, num_envs, ...).

    Compatible with MAPPO and other multi-agent algorithms.

    Example:
        ```python
        from memorax.environments.mettagrid import MettaGridWrapper
        from mettagrid.config.mettagrid_config import MettaGridConfig

        env = MettaGridWrapper(
            config=cfg,
            num_envs=8,
        )
        ```
    """

    def __init__(
        self,
        config: Any,  # MettaGridConfig
        num_envs: int,
        seed: int = 0,
    ):
        from mettagrid.mettagrid_c import (
            dtype_actions,
            dtype_observations,
            dtype_rewards,
            dtype_terminals,
            dtype_truncations,
        )
        from mettagrid.policy.policy_env_interface import PolicyEnvInterface
        from mettagrid.simulator import Simulator
        from mettagrid.simulator.simulator import Buffers

        self._config = config
        self._num_envs = num_envs
        self._seed = seed

        # Get env info
        policy_env_info = PolicyEnvInterface.from_mg_cfg(config)
        self._num_agents = config.game.num_agents
        self._obs_shape = policy_env_info.observation_space.shape  # (num_tokens, 3)
        self._obs_dtype = jnp.uint8
        self._num_actions = policy_env_info.action_space.n

        # Create simulators and simulations with their own buffers
        self._simulators = []
        self._simulations = []
        self._buffers_list = []

        for i in range(num_envs):
            # Create buffers for this env
            buffers = Buffers(
                observations=np.zeros(
                    (self._num_agents, *self._obs_shape),
                    dtype=dtype_observations,
                ),
                terminals=np.zeros(self._num_agents, dtype=dtype_terminals),
                truncations=np.zeros(self._num_agents, dtype=dtype_truncations),
                rewards=np.zeros(self._num_agents, dtype=dtype_rewards),
                masks=np.ones(self._num_agents, dtype=np.bool_),
                actions=np.zeros(self._num_agents, dtype=dtype_actions),
                teacher_actions=np.zeros(self._num_agents, dtype=dtype_actions),
            )
            self._buffers_list.append(buffers)

            # Create simulator and simulation
            simulator = Simulator()
            simulation = simulator.new_simulation(config, seed=seed + i, buffers=buffers)
            self._simulators.append(simulator)
            self._simulations.append(simulation)

        # Build action_spaces dict for MAPPO compatibility
        self._agent_ids = tuple(range(self._num_agents))
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

    def _reset_simulation(self, env_idx: int) -> None:
        """Reset a single simulation."""
        sim = self._simulations[env_idx]
        sim.close()
        self._simulations[env_idx] = self._simulators[env_idx].new_simulation(
            self._config,
            seed=self._seed + env_idx + np.random.randint(0, 10000),
            buffers=self._buffers_list[env_idx],
        )

    def reset(self, key: Key) -> Tuple[Array, MettaGridEnvState]:
        """Reset all environments.

        Returns:
            Tuple of (observations, state) where observations has shape
            (num_agents, num_envs, *obs_shape).
        """

        def _reset(key):
            for env_idx in range(self._num_envs):
                self._reset_simulation(env_idx)

            # Stack observations: (num_envs, num_agents, *obs_shape)
            obs = np.stack([buf.observations for buf in self._buffers_list], axis=0)
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

        state = MettaGridEnvState(step=0)
        return obs, state

    def step(
        self,
        key: Key,
        state: MettaGridEnvState,
        actions: Array,
    ) -> Tuple[Array, MettaGridEnvState, Array, Array, dict]:
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

            for env_idx in range(self._num_envs):
                buf = self._buffers_list[env_idx]
                sim = self._simulations[env_idx]

                # Set actions for this env (transpose to get per-env actions)
                np.copyto(buf.actions, actions[:, env_idx])

                # Step simulation
                sim.step()

                # Auto-reset if done
                if buf.terminals.all() or buf.truncations.all():
                    self._reset_simulation(env_idx)

            # Stack results: (num_envs, num_agents, ...)
            obs = np.stack([buf.observations for buf in self._buffers_list], axis=0)
            rewards = np.stack([buf.rewards for buf in self._buffers_list], axis=0)
            dones = np.stack(
                [buf.terminals | buf.truncations for buf in self._buffers_list], axis=0
            )

            # Transpose to (num_agents, num_envs, ...)
            obs = np.moveaxis(obs, 1, 0)
            rewards = np.moveaxis(rewards, 1, 0)
            dones = np.moveaxis(dones, 1, 0)

            return (
                jnp.array(obs, dtype=self._obs_dtype),
                jnp.array(rewards, dtype=jnp.float32),
                jnp.array(dones, dtype=jnp.bool_),
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

        new_state = MettaGridEnvState(step=state.step + 1)
        return obs, new_state, rewards, dones, {}

    def close(self) -> None:
        """Close all simulations."""
        for sim in self._simulations:
            sim.close()
        for simulator in self._simulators:
            simulator.close()
