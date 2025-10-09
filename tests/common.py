"""Shared testing utilities for the memory_rl algorithms test suite."""

from __future__ import annotations

from flax import struct
import jax
import jax.numpy as jnp
import optax
from flax import linen as nn
from flashbax.buffers import make_flat_buffer

from memory_rl.networks import Network
from memory_rl.networks.mlp import MLP
from memory_rl.networks.heads import (
    Alpha,
    Categorical,
    DiscreteQNetwork,
    Gaussian,
    VNetwork,
)


@struct.dataclass(frozen=True)
class DummyEnvState:
    """Minimal environment state that only tracks a timestep counter."""

    step: jnp.ndarray


@struct.dataclass(frozen=True)
class DummyEnvParams:
    """Hyper-parameters for the dummy environment used in smoke tests."""

    obs_dim: int = 4
    action_dim: int = 3
    max_steps_in_episode: int = 8


class DummyActionSpace:
    """Discrete action space exposing the sample API expected by algorithms."""

    def __init__(self, action_dim: int) -> None:
        self._action_dim = action_dim
        self.n = action_dim

    def sample(self, key: jax.Array) -> jax.Array:
        return jax.random.randint(
            key,
            shape=(),
            minval=0,
            maxval=self._action_dim,
            dtype=jnp.int32,
        )


class DummyEnv:
    """Stateless Gymnax-compatible environment for deterministic rollouts."""

    def reset(
        self, key: jax.Array, params: DummyEnvParams
    ) -> tuple[jax.Array, DummyEnvState]:
        del key
        obs = jnp.zeros((params.obs_dim,), dtype=jnp.float32)
        return obs, DummyEnvState(step=jnp.array(0, dtype=jnp.int32))

    def step(
        self,
        key: jax.Array,
        state: DummyEnvState,
        action: jax.Array,
        params: DummyEnvParams,
    ) -> tuple[jax.Array, DummyEnvState, jax.Array, jax.Array, dict[str, jax.Array]]:
        del key
        next_step = state.step + 1
        done = next_step >= params.max_steps_in_episode
        reward = jnp.where(done, 0.0, 1.0).astype(jnp.float32)
        action_cast = action.astype(jnp.float32)
        next_obs = jnp.broadcast_to(action_cast, (params.obs_dim,))
        info = {"step": next_step.astype(jnp.int32)}
        return (
            next_obs,
            DummyEnvState(step=next_step.astype(jnp.int32)),
            reward,
            done,
            info,
        )

    def action_space(self, params: DummyEnvParams) -> DummyActionSpace:
        return DummyActionSpace(params.action_dim)


@struct.dataclass(frozen=True)
class DummyContinuousEnvState:
    """Environment state for continuous-control smoke tests."""

    step: jnp.ndarray


@struct.dataclass(frozen=True)
class DummyContinuousEnvParams:
    """Parameters for the continuous dummy environment."""

    obs_dim: int = 4
    action_dim: int = 2
    max_steps_in_episode: int = 8
    action_low: float = -1.0
    action_high: float = 1.0


class DummyContinuousActionSpace:
    """Continuous action space with uniform sampling and shape metadata."""

    def __init__(self, action_dim: int, low: float, high: float) -> None:
        self.shape = (action_dim,)
        self.low = low
        self.high = high

    def sample(self, key: jax.Array) -> jax.Array:
        return jax.random.uniform(
            key,
            shape=self.shape,
            minval=self.low,
            maxval=self.high,
            dtype=jnp.float32,
        )


class DummyContinuousEnv:
    """Stateless environment emitting continuous observations and rewards."""

    def reset(
        self, key: jax.Array, params: DummyContinuousEnvParams
    ) -> tuple[jax.Array, DummyContinuousEnvState]:
        del key
        obs = jnp.zeros((params.obs_dim,), dtype=jnp.float32)
        return obs, DummyContinuousEnvState(step=jnp.array(0, dtype=jnp.int32))

    def step(
        self,
        key: jax.Array,
        state: DummyContinuousEnvState,
        action: jax.Array,
        params: DummyContinuousEnvParams,
    ) -> tuple[
        jax.Array, DummyContinuousEnvState, jax.Array, jax.Array, dict[str, jax.Array]
    ]:
        del key
        next_step = state.step + 1
        done = next_step >= params.max_steps_in_episode

        action = jnp.clip(action, params.action_low, params.action_high).astype(
            jnp.float32
        )
        reward = jnp.mean(action) * (1.0 - done.astype(jnp.float32))
        pad_size = max(0, params.obs_dim - params.action_dim)
        padding = jnp.zeros((pad_size,), dtype=jnp.float32)
        next_obs = jnp.concatenate([action, padding])[: params.obs_dim]
        info = {}

        return (
            next_obs.astype(jnp.float32),
            DummyContinuousEnvState(step=next_step.astype(jnp.int32)),
            reward.astype(jnp.float32),
            done.astype(jnp.bool_),
            info,
        )

    def action_space(
        self, params: DummyContinuousEnvParams
    ) -> DummyContinuousActionSpace:
        return DummyContinuousActionSpace(
            params.action_dim, params.action_low, params.action_high
        )


def make_discrete_network(obs_dim: int, action_dim: int) -> Network:
    """Create a lightweight feed-forward network for discrete control tests."""

    return Network(
        feature_extractor=MLP((32,)),
        torso=MLP((32,)),
        head=DiscreteQNetwork(action_dim=action_dim),
    )


def make_categorical_policy(obs_dim: int, action_dim: int) -> Network:
    """Build a policy network that produces categorical action distributions."""

    return Network(
        feature_extractor=MLP((32,)),
        torso=MLP((32,)),
        head=Categorical(action_dim=action_dim),
    )


def make_value_function(obs_dim: int) -> Network:
    """Construct a critic network for value estimation smoke tests."""

    return Network(
        feature_extractor=MLP((32,)),
        torso=MLP((32,)),
        head=VNetwork(),
    )


def make_buffer(num_envs: int, capacity: int, batch_size: int):
    """Factory for an episode buffer compatible with DQN-style sampling."""

    return make_flat_buffer(
        max_length=capacity,
        min_length=batch_size,
        sample_batch_size=batch_size,
        add_sequences=False,
        add_batch_size=num_envs,
    )


def make_optimizer(learning_rate: float) -> optax.GradientTransformation:
    """Create the shared Adam optimizer used in the smoke tests."""

    return optax.adam(learning_rate)


def make_linear_schedule(start: float, end: float, steps: int) -> optax.Schedule:
    """Provide a simple epsilon schedule for epsilon-greedy exploration."""

    steps = max(1, steps)
    return optax.linear_schedule(start, end, transition_steps=steps)


def make_gaussian_policy(obs_dim: int, action_dim: int) -> Network:
    """Policy network for continuous-control experiments."""

    return Network(
        feature_extractor=MLP((32,)),
        torso=MLP((32,)),
        head=Gaussian(action_dim=action_dim),
    )


class ContinuousTwinQNetwork(nn.Module):
    """Twin critic producing two scalar Q-values for continuous actions."""

    hidden_dim: int = 32

    @nn.compact
    def __call__(self, obs: jnp.ndarray, action: jnp.ndarray):
        x = jnp.concatenate([obs, action], axis=-1)
        x = nn.relu(nn.Dense(self.hidden_dim)(x))
        x = nn.relu(nn.Dense(self.hidden_dim)(x))
        q1 = nn.Dense(1)(x).squeeze(-1)
        q2 = nn.Dense(1)(x).squeeze(-1)
        return q1, q2


class DiscreteTwinQNetwork(nn.Module):
    """Twin critic for discrete actions returning logits per action."""

    action_dim: int
    hidden_dim: int = 32

    @nn.compact
    def __call__(self, obs: jnp.ndarray):
        x = nn.relu(nn.Dense(self.hidden_dim)(obs))
        x = nn.relu(nn.Dense(self.hidden_dim)(x))
        q1 = nn.Dense(self.action_dim)(x)
        q2 = nn.Dense(self.action_dim)(x)
        return q1, q2


def make_alpha_module(initial_alpha: float) -> Alpha:
    """Utility to build a learnable temperature parameter."""

    return Alpha(initial_alpha=initial_alpha)
