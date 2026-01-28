from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from flax import struct
from gymnax.environments.spaces import Box, Discrete

from memorax.utils.typing import Array, Key


@struct.dataclass
class MettagridState:
    step_count: Array


class FlattenObservationWrapper:
    """Wrapper that flattens observations from (num_agents, *obs_shape) to (num_agents, flat)."""

    def __init__(self, env):
        self._env = env
        self.obs_shape = (int(np.prod(env.obs_shape)),)

    def __getattr__(self, name):
        return getattr(self._env, name)

    def reset(self, key: Key):
        obs, state = self._env.reset(key)
        obs = obs.reshape(*obs.shape[:-len(self._env.obs_shape)], -1)
        return obs, state

    def step(self, key: Key, state, actions):
        obs, state, reward, done, info = self._env.step(key, state, actions)
        obs = obs.reshape(*obs.shape[:-len(self._env.obs_shape)], -1)
        return obs, state, reward, done, info


class MettagridEnvironment:
    """MettaGrid environment wrapper using PufferLib vectorization.

    Uses pufferlib.vector for efficient parallel environment execution,
    with jax.pure_callback and vmap_method="broadcast_all" for JAX integration.

    Must be used with jax.vmap - the vectorized env is created lazily based on
    the vmap batch size. PufferLib handles the actual parallelism internally.
    """

    def __init__(self, cfg, num_workers: int = 1):
        self.cfg = cfg
        self.num_workers = num_workers
        self.num_agents = cfg.game.num_agents
        self.agents = tuple(range(self.num_agents))

        self.environment = None
        self.num_envs = None
        self._initialize_spaces()

    def _make_env(self, buf=None, seed=0):
        from mettagrid import PufferMettaGridEnv, Simulator

        simulator = Simulator()
        return PufferMettaGridEnv(simulator=simulator, cfg=self.cfg, seed=seed, buf=buf)

    def _initialize_spaces(self):
        from mettagrid import PufferMettaGridEnv, Simulator

        simulator = Simulator()
        env = PufferMettaGridEnv(simulator=simulator, cfg=self.cfg, seed=0)
        obs, _ = env.reset(seed=0)

        _, *self.obs_shape = obs.shape
        self.action_dim = env.single_action_space.n
        self.action_spaces = {i: Discrete(self.action_dim) for i in self.agents}
        env.close()

        self._result_shape_dtype = (
            jax.ShapeDtypeStruct((self.num_agents, *self.obs_shape), jnp.float32),
            jax.ShapeDtypeStruct((self.num_agents,), jnp.float32),
            jax.ShapeDtypeStruct((self.num_agents,), jnp.bool_),
        )

    def _get_or_create_vec_env(self, num_envs: int):
        import pufferlib.vector

        if self.environment is not None and self.num_envs == num_envs:
            return self.environment

        if self.environment is not None:
            self.environment.close()

        backend = (
            pufferlib.vector.Multiprocessing
            if self.num_workers > 1
            else pufferlib.vector.Serial
        )

        self.environment = pufferlib.vector.make(
            self._make_env,
            num_envs=num_envs,
            num_workers=min(self.num_workers, num_envs),
            backend=backend,
        )
        self.num_envs = num_envs
        return self.environment

    def reset(self, key: Key) -> tuple[Array, MettagridState]:
        def callback(key_data):
            key_data = np.asarray(key_data)
            batch_shape = key_data.shape[:-1]
            num_envs = int(np.prod(batch_shape))

            seed = int(key_data.flat[0])
            vec_env = self._get_or_create_vec_env(num_envs)
            obs, _ = vec_env.reset(seed=seed)

            obs = obs.astype(np.float32).reshape(
                *batch_shape, self.num_agents, *self.obs_shape
            )
            reward = np.zeros((*batch_shape, self.num_agents), dtype=np.float32)
            done = np.zeros((*batch_shape, self.num_agents), dtype=np.bool_)

            return obs, reward, done

        obs, _, _ = jax.pure_callback(
            callback,
            self._result_shape_dtype,
            jax.random.key_data(key),
            vmap_method="broadcast_all",
        )
        return obs, MettagridState(step_count=jnp.zeros(()))

    def step(
        self, key: Key, state: MettagridState, actions: Array
    ) -> tuple[Array, MettagridState, Array, Array, dict[str, Any]]:
        del key

        def callback(actions):
            actions = np.asarray(actions, dtype=np.int32)
            batch_shape = actions.shape[:-1]
            num_envs = int(np.prod(batch_shape))

            actions_flat = actions.reshape(-1)
            vec_env = self._get_or_create_vec_env(num_envs)

            vec_env.send(actions_flat)
            obs, reward, terminated, truncated, _, _, _ = vec_env.recv()

            obs = obs.astype(np.float32).reshape(
                *batch_shape, self.num_agents, *self.obs_shape
            )
            reward = reward.astype(np.float32).reshape(*batch_shape, self.num_agents)
            done = (terminated | truncated).reshape(*batch_shape, self.num_agents)

            return obs, reward, done

        obs, reward, done = jax.pure_callback(
            callback,
            self._result_shape_dtype,
            actions,
            vmap_method="broadcast_all",
        )
        new_state = MettagridState(step_count=state.step_count + 1)
        return obs, new_state, reward, done, {}

    def close(self):
        if self.environment is not None:
            self.environment.close()
            self.environment = None
