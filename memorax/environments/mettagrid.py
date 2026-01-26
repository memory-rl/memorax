from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from flax import struct
from gymnax.environments.spaces import Discrete
from jax import Array
from jax.random import PRNGKey as Key

envs_mettagrid: dict[int, Any] = {}


@struct.dataclass
class MettagridState:
    env_id: Array


class MettagridEnvironment:
    """Mettagrid environment wrapper compatible with IPPO algorithm.

    Provides a JAX-compatible interface for the Mettagrid multi-agent environment.
    """

    def __init__(self, id: str, num_agents: int = 24, **kwargs):
        from mettagrid.builder.envs import make_arena
        from mettagrid.envs.mettagrid_puffer_env import MettaGridPufferEnv
        from mettagrid.simulator import Simulator

        self.id = id
        self.mettagrid_id = id.split("/", 1)[1]
        self._num_agents = num_agents
        self.kwargs = kwargs

        simulator = Simulator()
        cfg = make_arena(num_agents=num_agents, **kwargs)
        env = MettaGridPufferEnv(simulator=simulator, cfg=cfg)

        obs, _ = env.reset(seed=0)
        action = np.zeros(env.num_agents, dtype=np.int32)
        obs, reward, terminated, truncated, info = env.step(action)
        env.close()

        self._agents = tuple(range(num_agents))
        self._action_spaces = {i: Discrete(env.single_action_space.n) for i in self._agents}
        self._observation_space = env.single_observation_space
        self._obs_shape = obs.shape[1:]

        env_state = MettagridState(env_id=jnp.array([0], dtype=jnp.int32))
        sample_obs = jnp.array(obs)[None]
        sample_reward = jnp.array(reward, dtype=jnp.float32)[None]
        sample_done = jnp.array(terminated, dtype=jnp.bool)[None]
        self.return_shape_dtype = jax.tree.map(
            lambda x: jax.ShapeDtypeStruct(x.shape[1:], x.dtype),
            (sample_obs, env_state, sample_reward, sample_done, {}),
        )

    @property
    def agents(self):
        return self._agents

    @property
    def num_agents(self):
        return self._num_agents

    @property
    def action_spaces(self):
        return self._action_spaces

    def init(self, key: Key) -> tuple[Array, MettagridState]:
        """Initialize environments and return initial observations and state.

        This method creates new environment instances and should be called once
        at the start of training. Use reset() for subsequent episode resets.

        Args:
            key: JAX random key for seeding environments.

        Returns:
            Tuple of (obs, env_state) where obs has shape (num_agents, *obs_shape).
        """
        from mettagrid.builder.envs import make_arena
        from mettagrid.envs.mettagrid_puffer_env import MettaGridPufferEnv
        from mettagrid.simulator import Simulator

        def callback(key):
            global envs_mettagrid
            shape = key.shape[:-1]
            keys_flat = np.reshape(np.asarray(key), (-1, key.shape[-1]))
            num_envs = keys_flat.shape[0]

            all_obs = []
            env_ids = []
            for i in range(num_envs):
                seed = int(keys_flat[i, 0])
                simulator = Simulator()
                cfg = make_arena(num_agents=self._num_agents, **self.kwargs)
                env = MettaGridPufferEnv(simulator=simulator, cfg=cfg, seed=seed)
                obs, _ = env.reset(seed=seed)

                env_id = len(envs_mettagrid)
                envs_mettagrid[env_id] = env
                env_ids.append(env_id)
                all_obs.append(obs)

            obs_stacked = np.stack(all_obs, axis=0)

            obs = jnp.reshape(jnp.array(obs_stacked), shape + obs_stacked.shape[1:])
            env_state = MettagridState(
                env_id=jnp.reshape(jnp.array(env_ids, dtype=jnp.int32), shape)
            )
            reward = jnp.zeros(shape + (self._num_agents,), dtype=jnp.float32)
            done = jnp.zeros(shape + (self._num_agents,), dtype=jnp.bool)
            return obs, env_state, reward, done, {}

        obs, env_state, _, _, _ = jax.pure_callback(
            callback,
            self.return_shape_dtype,
            jax.random.key_data(key),
            vmap_method="broadcast_all",
        )
        return obs, env_state

    def reset(self, key: Key) -> tuple[Array, MettagridState]:
        """Reset environments by creating new instances.

        Creates new environment instances for each call. This is compatible with
        the IPPO algorithm which calls reset(key) without prior state.

        Args:
            key: JAX random key for seeding environments.

        Returns:
            Tuple of (obs, env_state) where obs has shape (num_agents, *obs_shape).
        """
        from mettagrid.builder.envs import make_arena
        from mettagrid.envs.mettagrid_puffer_env import MettaGridPufferEnv
        from mettagrid.simulator import Simulator

        def callback(key):
            global envs_mettagrid
            shape = key.shape[:-1]
            keys_flat = np.reshape(np.asarray(key), (-1, key.shape[-1]))
            num_envs = keys_flat.shape[0]

            all_obs = []
            env_ids = []
            for i in range(num_envs):
                seed = int(keys_flat[i, 0])
                simulator = Simulator()
                cfg = make_arena(num_agents=self._num_agents, **self.kwargs)
                env = MettaGridPufferEnv(simulator=simulator, cfg=cfg, seed=seed)
                obs, _ = env.reset(seed=seed)

                env_id = len(envs_mettagrid)
                envs_mettagrid[env_id] = env
                env_ids.append(env_id)
                all_obs.append(obs)

            obs_stacked = np.stack(all_obs, axis=0)

            obs = jnp.reshape(jnp.array(obs_stacked), shape + obs_stacked.shape[1:])
            env_state = MettagridState(
                env_id=jnp.reshape(jnp.array(env_ids, dtype=jnp.int32), shape)
            )
            reward = jnp.zeros(shape + (self._num_agents,), dtype=jnp.float32)
            done = jnp.zeros(shape + (self._num_agents,), dtype=jnp.bool)
            return obs, env_state, reward, done, {}

        obs, env_state, _, _, _ = jax.pure_callback(
            callback,
            self.return_shape_dtype,
            jax.random.key_data(key),
            vmap_method="broadcast_all",
        )
        return obs, env_state

    def step(
        self, key: Key, env_state: MettagridState, actions: Array
    ) -> tuple[Array, MettagridState, Array, Array, dict]:
        """Step the environments forward.

        Args:
            key: JAX random key (unused, kept for API compatibility).
            env_state: Current environment state containing env_ids.
            actions: Actions array with shape (num_agents,) per environment.

        Returns:
            Tuple of (next_obs, env_state, reward, done, info) where:
                - next_obs: Shape (num_agents, *obs_shape)
                - reward: Shape (num_agents,)
                - done: Shape (num_agents,), boolean
                - info: Dictionary with additional information
        """
        del key

        def callback(env_id, action):
            global envs_mettagrid
            shape = env_id.shape
            env_ids_flat = np.ravel(np.asarray(env_id))
            num_envs = env_ids_flat.shape[0]

            actions_np = np.asarray(action, dtype=np.int32)
            if actions_np.ndim == 1:
                actions_np = actions_np[None, :]
            else:
                actions_np = np.reshape(actions_np, (num_envs, -1))

            all_obs = []
            all_rewards = []
            all_done = []

            for i in range(num_envs):
                env = envs_mettagrid[int(env_ids_flat[i])]
                obs, reward, terminated, truncated, info = env.step(actions_np[i])
                all_obs.append(obs)
                all_rewards.append(reward)
                all_done.append(terminated | truncated)

            obs_stacked = np.stack(all_obs, axis=0)
            reward_stacked = np.stack(all_rewards, axis=0)
            done_stacked = np.stack(all_done, axis=0)

            obs = jnp.reshape(jnp.array(obs_stacked), shape + obs_stacked.shape[1:])
            env_state = MettagridState(env_id=env_id)
            reward = jnp.reshape(
                jnp.array(reward_stacked, dtype=jnp.float32),
                shape + reward_stacked.shape[1:],
            )
            done = jnp.reshape(
                jnp.array(done_stacked, dtype=jnp.bool),
                shape + done_stacked.shape[1:],
            )
            return obs, env_state, reward, done, {}

        obs, env_state, reward, done, info = jax.pure_callback(
            callback,
            self.return_shape_dtype,
            env_state.env_id,
            actions,
            vmap_method="broadcast_all",
        )
        return obs, env_state, reward, done, info

