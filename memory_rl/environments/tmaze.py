from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
from flax import struct
from gymnax.environments import environment, spaces


@struct.dataclass
class EnvState(environment.EnvState):
    """Immutable TMaze state."""

    x: jax.Array  # corridor coordinate
    y: jax.Array  # lateral coordinate (−1, 0, +1)
    goal_y: jax.Array  # hidden goal direction
    oracle_visited: jax.Array  # bool flag


@struct.dataclass
class EnvParams(environment.EnvParams):
    """Configurable TMaze parameters."""

    corridor_length: int = 10
    oracle_length: int = 0
    episode_length: int = 11  # corridor_length + 1 for classic passive
    goal_reward: float = 1.0
    penalty: float = 0.0
    distract_reward: float = 0.0

    add_timestep: bool = struct.field(pytree_node=False, default=False)
    ambiguous_position: bool = struct.field(pytree_node=False, default=True)
    expose_goal: bool = struct.field(pytree_node=False, default=False)


class TMaze(environment.Environment[EnvState, EnvParams]):
    """Pure-JAX TMaze."""

    ACTION_MAP = jnp.array([[1, 0], [0, 1], [-1, 0], [0, -1]], dtype=jnp.int32)

    @staticmethod
    def _obs_dim(p: EnvParams) -> int:
        """All variants emit 3 floats (+1 for optional timestep)."""
        return 3 + int(p.add_timestep)

    @staticmethod
    def _valid_position(x: jax.Array, y: jax.Array, p: EnvParams) -> jax.Array:
        """True if (x,y) lies on the TMaze graph."""
        max_x = p.corridor_length + p.oracle_length
        corridor = (y == 0) & (x >= 0) & (x <= max_x)
        goals = (x == max_x) & ((y == 1) | (y == -1))
        return corridor | goals

    def __init__(self):
        super().__init__()
        self.obs_shape = (self._obs_dim(self.default_params),)

    @property
    def default_params(self) -> EnvParams:  # overridden by subclasses
        return EnvParams()

    @property
    def num_actions(self) -> int:
        return 4

    def reset_env(
        self, key: jax.Array, params: EnvParams
    ) -> tuple[jax.Array, EnvState]:
        key, sub = jax.random.split(key)
        goal_y = jnp.where(jax.random.bernoulli(sub), 1, -1).astype(jnp.int32)
        state = EnvState(
            x=jnp.asarray(params.oracle_length, jnp.int32),
            y=jnp.asarray(0, jnp.int32),
            goal_y=goal_y,
            oracle_visited=jnp.asarray(False),
            time=0,
        )
        return self.get_obs(state, params), state

    def step_env(
        self,
        key,
        state,
        action,
        params,
    ):
        move = self.ACTION_MAP[action]
        cand_x = state.x + move[0]
        cand_y = state.y + move[1]
        valid = self._valid_position(cand_x, cand_y, params)
        new_x = jnp.where(valid, cand_x, state.x)
        new_y = jnp.where(valid, cand_y, state.y)

        new_time = state.time + 1
        new_oracle_visited = state.oracle_visited | (new_x == 0)
        done = new_time >= params.episode_length

        step_penalty = jnp.where(
            new_x < (new_time - params.oracle_length), params.penalty, 0.0
        )
        distract = jnp.where(new_x == 0, params.distract_reward, 0.0)
        step_reward = step_penalty + distract
        goal_bonus = jnp.where((new_y == state.goal_y) & done, params.goal_reward, 0.0)
        reward = jnp.where(done, goal_bonus, step_reward).astype(jnp.float32)

        next_state = EnvState(
            x=new_x,
            y=new_y,
            goal_y=state.goal_y,
            oracle_visited=new_oracle_visited,
            time=new_time,
        )

        return (
            jax.lax.stop_gradient(self.get_obs(next_state, params)),
            jax.lax.stop_gradient(next_state),
            reward,
            done,
            {"discount": self.discount(next_state, params)},
        )

    def get_obs(self, state, params, key=None) -> jax.Array:
        """Returns float32 observation of length 3 (+t if requested)."""
        x_i, y_i, g_i = state.x, state.y, state.goal_y
        oracle_exp = jnp.where((x_i == 0) & (~state.oracle_visited), g_i, 0)

        at_t = x_i >= (params.oracle_length + params.corridor_length)
        first = jnp.where(at_t, 1.0, 0.0)
        second = jnp.where(
            x_i == 0,
            oracle_exp.astype(jnp.float32),
            jnp.where(at_t, y_i.astype(jnp.float32), 0.0),
        )
        ambig_obs = jnp.array([first, second, 0.0], dtype=jnp.float32)

        third = jnp.where(
            params.expose_goal,
            jnp.where(state.oracle_visited, g_i, 0),
            jnp.where(x_i == 0, oracle_exp, 0),
        ).astype(jnp.float32)
        markov_obs = jnp.array(
            [x_i.astype(jnp.float32), y_i.astype(jnp.float32), third], dtype=jnp.float32
        )

        base_obs = jax.lax.select(
            jnp.asarray(params.ambiguous_position), ambig_obs, markov_obs
        )

        if params.add_timestep:
            base_obs = jnp.concatenate(
                [base_obs, jnp.array([state.time], dtype=jnp.float32)]
            )

        return base_obs

    def is_terminal(self, state: EnvState, params: EnvParams) -> jax.Array:
        return state.time >= params.episode_length

    def discount(self, state: EnvState, params: EnvParams) -> jax.Array:
        return jnp.where(self.is_terminal(state, params), 0.0, 1.0)

    def action_space(self, params: EnvParams | None = None) -> spaces.Discrete:
        return spaces.Discrete(self.num_actions)

    def observation_space(self, params: EnvParams) -> spaces.Box:
        dim = self._obs_dim(params)
        high = jnp.ones((dim,), dtype=jnp.float32)
        return spaces.Box(-high, high, (dim,), dtype=jnp.float32)

    def state_space(self, params: EnvParams) -> spaces.Dict:
        max_x = params.corridor_length + params.oracle_length
        return spaces.Dict(
            {
                "x": spaces.Box(0, max_x, (), jnp.int32),
                "y": spaces.Box(-1, 1, (), jnp.int32),
                "goal_y": spaces.Box(-1, 1, (), jnp.int32),
                "oracle_visited": spaces.Discrete(2),
                "time": spaces.Discrete(params.episode_length + 1),
            }
        )

    @property
    def name(self) -> str:
        return "TMaze"


class TMazeClassicPassive(TMaze):
    """Passive-memory TMaze (oracle at start, no explicit exposure after step 0)."""

    @property
    def default_params(self) -> EnvParams:
        cl = 10
        return EnvParams(
            corridor_length=cl,
            oracle_length=0,
            episode_length=cl + 1,
            ambiguous_position=True,
        )


class TMazeClassicActive(TMaze):
    """Active-memory TMaze with oracle one step in front of start."""

    @property
    def default_params(self) -> EnvParams:
        cl, ol = 10, 1
        return EnvParams(
            corridor_length=cl,
            oracle_length=ol,
            episode_length=cl + 2 * ol + 1,
            ambiguous_position=True,
        )
