import jax
import jax.numpy as jnp
import gymnax
import navix as nx
from navix.entities import Player, Goal
from navix.spaces import Discrete, Continuous
from flax import struct
from functools import partial

from memory_rl.utils import NavixGymnaxWrapper


@struct.dataclass
class TMazeState:
    t: int
    x: int
    y: int
    goal_side: int
    oracle_seen: bool
    key: any = None
    cache: any = None


class TMazeEnv(nx.environments.Environment):
    """TMaze environment (passive/active) as in 'Where Do Transformers Shine'."""

    def __init__(self, L=5, active=False, seed=0):
        # L: corridor length (from O to J)
        # Actions: 0=L, 1=R, 2=U, 3=D
        # States: x (horizontal, 0=O, L=J), y (vertical, 0=middle, -1=up, 1=down)
        # G1: (L, -1), G2: (L, 1)
        key = jax.random.PRNGKey(seed)
        # Passive: O=S=(0,0), Active: O=(-1,0), S=(0,0)
        if active:
            start_x = 0
            oracle_x = -1
        else:
            start_x = 0
            oracle_x = 0
        start_y = 0
        L = int(L)
        width = L - oracle_x + 1  # x in [-1, L]
        height = 3  # y in {-1, 0, 1}
        G1 = (L, -1)
        G2 = (L, 1)
        J = (L, 0)
        O = (oracle_x, 0)

        def reward_fn(prev_state, action, state):
            at_goal = (state.x == L) & (state.y == state.goal_side)
            step_penalty = ((state.x) < (state.t - active)) * -1 / L
            return jnp.where(at_goal, 1.0, step_penalty)

        def termination_fn(prev_state, action, state):
            # Terminal if reached goal states
            goal_reached = (state.x == L) & ((state.y == -1) | (state.y == 1))
            return goal_reached

        def observation_fn(state):
            # obs_type: 0=null, 1=oracle, 2=junction, 3=goal
            is_oracle = (state.x == oracle_x) & (state.y == 0)
            is_junction = (state.x == L) & (state.y == 0)
            is_goal = (state.x == L) & ((state.y == -1) | (state.y == 1))
            obs_type = jnp.where(
                is_goal, 3, jnp.where(is_junction, 2, jnp.where(is_oracle, 1, 0))
            )
            # goal_obs: only revealed once per episode when oracle is first visited
            if active:
                # Only reveal goal_side if currently at oracle AND oracle hasn't been seen before
                at_oracle_first_time = is_oracle & (~state.oracle_seen)
                goal_obs = jnp.where(at_oracle_first_time, state.goal_side, -2)
            else:
                # Passive: reveal at start position (0,0) - this is the oracle position for passive
                at_start_oracle = (state.x == 0) & (state.y == 0) & (~state.oracle_seen)
                goal_obs = jnp.where(at_start_oracle, state.goal_side, -2)
            return jnp.array([obs_type, goal_obs], dtype=jnp.int32)

        super().__init__(
            height=height,
            width=width,
            max_steps=L + 1 if not active else L + 3,
            gamma=0.99,
            penality_coeff=0.0,
            observation_fn=observation_fn,
            reward_fn=reward_fn,
            termination_fn=termination_fn,
            # Note: Using Continuous space with int32 dtype for discrete obs since navix lacks MultiDiscrete
            observation_space=Continuous.create(
                shape=(2,),
                minimum=jnp.asarray([0, -2]),
                maximum=jnp.asarray([3, 1]),
                dtype=jnp.int32,
            ),
            action_space=Discrete.create(4),  # 0=L, 1=R, 2=U, 3=D
            reward_space=Continuous.create(
                shape=(), minimum=jnp.asarray(-1.0), maximum=jnp.asarray(1.0)
            ),
        )

        def _reset(self, key, cache=None):
            key, sub = jax.random.split(key)
            goal_side = jnp.where(
                jax.random.bernoulli(sub), jnp.int32(1), jnp.int32(-1)
            )
            state = TMazeState(
                t=0,
                x=start_x,
                y=start_y,
                goal_side=goal_side,
                oracle_seen=False,  # Oracle not seen yet for both passive and active
                key=key,
                cache=cache,
            )
            timestep = nx.environments.Timestep(
                t=jnp.asarray(0),
                observation=observation_fn(state),
                action=jnp.asarray(0),
                reward=jnp.asarray(0.0),
                step_type=jnp.asarray(0),
                state=state,
                info={"return": jnp.asarray(0.0)},
            )
            return timestep

        def _step(self, timestep, action):
            x, y = timestep.state.x, timestep.state.y
            # Move
            dx = jnp.where(action == 0, -1, jnp.where(action == 1, 1, 0))
            dy = jnp.where(action == 2, -1, jnp.where(action == 3, 1, 0))
            # Only allow up/down at junction (x==L)
            can_up_down = x == L
            dy = jnp.where(can_up_down, dy, 0)
            # Next state, enforce boundaries
            nx_ = jnp.clip(x + dx, oracle_x, L)
            ny_ = jnp.clip(y + dy, -1, 1)
            # Create intermediate state for observation (before marking oracle as seen)
            state_for_obs = TMazeState(
                t=timestep.state.t + 1,
                x=nx_,
                y=ny_,
                goal_side=timestep.state.goal_side,
                oracle_seen=timestep.state.oracle_seen,  # Keep previous oracle_seen for observation
                key=timestep.state.key,
                cache=timestep.state.cache,
            )

            # Mark oracle as seen when visiting oracle position
            oracle_seen = timestep.state.oracle_seen | ((nx_ == oracle_x) & (ny_ == 0))
            state = TMazeState(
                t=timestep.state.t + 1,
                x=nx_,
                y=ny_,
                goal_side=timestep.state.goal_side,
                oracle_seen=oracle_seen,
                key=timestep.state.key,
                cache=timestep.state.cache,
            )
            reward = reward_fn(timestep.state, action, state)
            done = termination_fn(timestep.state, action, state)
            # Check time horizon: T = L + 1 (passive) or L + 3 (active)
            max_time = L + 1 if not active else L + 3
            time_exceeded = (timestep.t + 1) >= max_time
            # Episode ends if goal reached OR time limit exceeded
            episode_end = done | time_exceeded
            step_type = jnp.where(episode_end, 2, 0)
            new_timestep = nx.environments.Timestep(
                t=timestep.t + 1,
                observation=observation_fn(
                    state_for_obs
                ),  # Use state before oracle_seen update
                action=action,
                reward=reward,
                step_type=step_type,
                state=state,
                info={"return": jnp.asarray(0.0)},
            )
            return new_timestep

        object.__setattr__(self, "_reset", _reset.__get__(self))
        object.__setattr__(self, "_step", _step.__get__(self))


def make_tmaze_env(env_id=None, L=5, active=False, seed=0):
    env = TMazeEnv(L=L, active=active, seed=seed)
    max_steps = env.max_steps
    env = NavixGymnaxWrapper(env=env)
    env_params = gymnax.environments.environment.EnvParams(
        max_steps_in_episode=max_steps
    )
    return env, env_params


# if __name__ == "__main__":
#     env, env_params = make_tmaze_env(L=5, active=False, seed=0)
#
#     timestep = env.reset(env_params)
#     env._env.visualize(timestep)
