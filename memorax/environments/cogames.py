"""CoGames environment registration for PufferLib."""

from functools import partial
from typing import Optional, Sequence, Tuple

import jax.numpy as jnp
import numpy as np
import pufferlib.vector
from cogames.cli.mission import get_mission
from gymnax.environments import spaces
from mettagrid.envs.mettagrid_puffer_env import MettaGridPufferEnv
from mettagrid.simulator import Simulator

from memorax.environments.pufferlib import (
    PufferLibEnvState,
    PufferLibMultiAgentWrapper,
    PufferLibWrapper,
    register,
)
from memorax.utils.typing import Array, Key


class BoxObsWrapper:
    """Converts token-based CoGames observations to spatial grid format.

    Token format per triplet: (coord_byte, attr_idx, attr_val).
    coord_byte encodes x = coord & 0x0F, y = coord >> 4.
    Invalid tokens (coord == 0xFF) are masked out.
    Output grid shape: (H, W, num_layers) with float32 values normalized by /255.
    """

    def __init__(self, env, grid_h: int = 11, grid_w: int = 11, num_layers: int = 24):
        self._env = env
        self._grid_h = grid_h
        self._grid_w = grid_w
        self._num_layers = num_layers
        self._grid_shape = (grid_h, grid_w, num_layers)
        import math
        self._token_obs_size = math.prod(env._obs_shape)

    def _tokens_to_grid(self, tokens: jnp.ndarray) -> jnp.ndarray:
        """Convert flat token array to spatial grid using JAX ops (GPU).

        Args:
            tokens: int32 array of shape (*batch, num_tokens * 3).

        Returns:
            float32 array of shape (*batch, H, W, num_layers).
        """
        batch_shape = tokens.shape[:-1]
        flat = tokens.reshape(-1, tokens.shape[-1])

        num_samples = flat.shape[0]
        num_triplets = flat.shape[-1] // 3

        triplets = flat.reshape(num_samples, num_triplets, 3)
        coords = triplets[:, :, 0]
        attr_idx = triplets[:, :, 1]
        attr_val = triplets[:, :, 2]

        x = coords & 0x0F
        y = coords >> 4
        layer = attr_idx.astype(jnp.int32)

        valid = (
            (coords != 0xFF)
            & (x < self._grid_w)
            & (y < self._grid_h)
            & (layer < self._num_layers)
        )

        # Out-of-bounds indices for invalid triplets (ignored by mode='drop')
        x = jnp.where(valid, x, self._grid_w)
        y = jnp.where(valid, y, self._grid_h)
        layer = jnp.where(valid, layer, self._num_layers)

        grids = jnp.zeros(
            (num_samples, self._grid_h, self._grid_w, self._num_layers),
            dtype=jnp.float32,
        )

        sample_idx = jnp.broadcast_to(
            jnp.arange(num_samples)[:, None], (num_samples, num_triplets)
        )
        values = attr_val.astype(jnp.float32) / 255.0
        grids = grids.at[sample_idx, y, x, layer].set(values, mode="drop")

        return grids.reshape(*batch_shape, *self._grid_shape)

    @property
    def obs_shape(self) -> tuple:
        return self._grid_shape

    @property
    def obs_dtype(self):
        return jnp.float32

    @property
    def num_envs(self) -> int:
        return self._env.num_envs

    @property
    def agents(self) -> tuple:
        return self._env.agents

    @property
    def num_agents(self) -> int:
        return self._env.num_agents

    @property
    def action_spaces(self) -> dict:
        return self._env.action_spaces

    @property
    def num_actions(self) -> int:
        return self._env.num_actions

    @property
    def vectorized(self) -> bool:
        return self._env.vectorized

    def reset(self, key: Key) -> Tuple[Array, PufferLibEnvState]:
        import jax

        def _reset(key):
            obs, _ = self._env._env.reset()
            obs = obs.reshape(self._env._num_envs, self._env._num_agents, -1)
            obs = np.moveaxis(obs, 1, 0)
            return jnp.array(obs, dtype=jnp.int32)

        token_shape = (self._env._num_agents, self._env._num_envs, self._token_obs_size)

        tokens = jax.pure_callback(
            _reset,
            jax.ShapeDtypeStruct(token_shape, jnp.int32),
            key,
            vmap_method="broadcast_all",
        )
        obs = self._tokens_to_grid(tokens)

        state = PufferLibEnvState(step=0)
        return obs, state

    def step(
        self,
        key: Key,
        state: PufferLibEnvState,
        actions: Array,
    ) -> Tuple[Array, PufferLibEnvState, Array, Array, dict]:
        import jax

        def _step(actions):
            actions_np = np.asarray(actions, dtype=np.int32)
            actions_np = np.moveaxis(actions_np, 0, 1).flatten()

            obs, rewards, dones, truncs, infos = self._env._env.step(actions_np)

            obs = obs.reshape(self._env._num_envs, self._env._num_agents, -1)
            obs = np.moveaxis(obs, 1, 0)

            rewards = rewards.reshape(self._env._num_envs, self._env._num_agents)
            rewards = np.moveaxis(rewards, 1, 0)

            combined_dones = dones | truncs
            combined_dones = combined_dones.reshape(
                self._env._num_envs, self._env._num_agents
            )
            combined_dones = np.moveaxis(combined_dones, 1, 0)

            return (
                jnp.array(obs, dtype=jnp.int32),
                jnp.array(rewards, dtype=jnp.float32),
                jnp.array(combined_dones, dtype=jnp.bool_),
            )

        token_shape = (self._env._num_agents, self._env._num_envs, self._token_obs_size)
        scalar_shape = (self._env._num_agents, self._env._num_envs)

        tokens, rewards, dones = jax.pure_callback(
            _step,
            (
                jax.ShapeDtypeStruct(token_shape, jnp.int32),
                jax.ShapeDtypeStruct(scalar_shape, jnp.float32),
                jax.ShapeDtypeStruct(scalar_shape, jnp.bool_),
            ),
            actions,
            vmap_method="broadcast_all",
        )
        obs = self._tokens_to_grid(tokens)

        new_state = PufferLibEnvState(step=state.step + 1)
        return obs, new_state, rewards, dones, {}

    def observation_space(self, params=None) -> spaces.Box:
        return spaces.Box(
            low=0.0,
            high=1.0,
            shape=self._grid_shape,
            dtype=jnp.float32,
        )

    def action_space(self, params=None) -> spaces.Discrete:
        return spaces.Discrete(self.num_actions)


def _make_single(env_id: str, variants: Optional[Sequence[str]] = None, **kwargs):
    """Create a single MettaGrid environment."""
    _, cfg, _ = get_mission(env_id)

    if variants:
        try:
            from cogames.cogs_vs_clips.cogsguard_reward_variants import (
                apply_reward_variants,
            )

            apply_reward_variants(cfg, variants=variants)
        except ImportError:
            pass

    simulator = Simulator()
    return MettaGridPufferEnv(simulator, cfg, **kwargs)


def _cogames_factory(
    env_id: str,
    num_envs: int,
    variants: Optional[Sequence[str]] = None,
    multi_agent: bool = False,
    backend=pufferlib.vector.Serial,
    **kwargs,
):
    """Factory registered under the 'cogames' namespace."""
    _, cfg, _ = get_mission(env_id)
    num_agents = cfg.game.num_agents

    puffer_env = pufferlib.vector.make(
        partial(_make_single, variants=variants),
        num_envs=num_envs,
        backend=backend,
        env_kwargs={"env_id": env_id},
    )

    if multi_agent:
        wrapper = PufferLibMultiAgentWrapper(puffer_env, num_agents=num_agents)
    else:
        wrapper = PufferLibWrapper(puffer_env)

    env_info = {
        "agents_per_env": num_agents,
    }

    return wrapper, env_info


register("cogames", _cogames_factory)
