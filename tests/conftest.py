"""Pytest fixtures shared across algorithm smoke tests."""

from __future__ import annotations

import pytest
import jax

from tests.common import (
    DummyContinuousEnv,
    DummyContinuousEnvParams,
    DummyEnv,
    DummyEnvParams,
    ContinuousTwinQNetwork,
    DiscreteTwinQNetwork,
    make_buffer,
    make_alpha_module,
    make_categorical_policy,
    make_discrete_network,
    make_linear_schedule,
    make_optimizer,
    make_gaussian_policy,
    make_value_function,
)


@pytest.fixture(scope="session")
def dummy_env() -> DummyEnv:
    return DummyEnv()


@pytest.fixture(scope="session")
def dummy_env_params() -> DummyEnvParams:
    return DummyEnvParams()


@pytest.fixture(scope="session")
def dummy_continuous_env() -> DummyContinuousEnv:
    return DummyContinuousEnv()


@pytest.fixture(scope="session")
def dummy_continuous_env_params() -> DummyContinuousEnvParams:
    return DummyContinuousEnvParams()


@pytest.fixture()
def rng_key() -> jax.Array:
    return jax.random.key(0)


@pytest.fixture()
def dqn_components(dummy_env_params: DummyEnvParams):
    buffer = make_buffer(num_envs=2, capacity=16, batch_size=4)
    q_network = make_discrete_network(
        obs_dim=dummy_env_params.obs_dim,
        action_dim=dummy_env_params.action_dim,
    )
    optimizer = make_optimizer(learning_rate=3e-4)
    schedule = make_linear_schedule(start=1.0, end=0.05, steps=100)
    return buffer, q_network, optimizer, schedule


@pytest.fixture()
def ppo_components(dummy_env_params: DummyEnvParams):
    actor = make_categorical_policy(
        obs_dim=dummy_env_params.obs_dim,
        action_dim=dummy_env_params.action_dim,
    )
    critic = make_value_function(obs_dim=dummy_env_params.obs_dim)
    optimizer = make_optimizer(learning_rate=1e-3)
    return actor, critic, optimizer


@pytest.fixture()
def ppo_continuous_components(dummy_continuous_env_params: DummyContinuousEnvParams):
    actor = make_gaussian_policy(
        obs_dim=dummy_continuous_env_params.obs_dim,
        action_dim=dummy_continuous_env_params.action_dim,
    )
    critic = make_value_function(obs_dim=dummy_continuous_env_params.obs_dim)
    optimizer = make_optimizer(learning_rate=1e-3)
    return actor, critic, optimizer


@pytest.fixture()
def sac_components(dummy_continuous_env_params: DummyContinuousEnvParams):
    actor = make_gaussian_policy(
        obs_dim=dummy_continuous_env_params.obs_dim,
        action_dim=dummy_continuous_env_params.action_dim,
    )
    critic = ContinuousTwinQNetwork()
    buffer = make_buffer(num_envs=2, capacity=32, batch_size=4)
    alpha_network = make_alpha_module(initial_alpha=0.1)
    actor_optimizer = make_optimizer(learning_rate=3e-4)
    critic_optimizer = make_optimizer(learning_rate=3e-4)
    alpha_optimizer = make_optimizer(learning_rate=1e-3)
    return (
        buffer,
        actor,
        critic,
        alpha_network,
        actor_optimizer,
        critic_optimizer,
        alpha_optimizer,
    )


@pytest.fixture()
def sacd_components(dummy_env_params: DummyEnvParams):
    actor = make_categorical_policy(
        obs_dim=dummy_env_params.obs_dim,
        action_dim=dummy_env_params.action_dim,
    )
    critic = DiscreteTwinQNetwork(action_dim=dummy_env_params.action_dim)
    buffer = make_buffer(num_envs=2, capacity=32, batch_size=4)
    alpha_network = make_alpha_module(initial_alpha=0.1)
    actor_optimizer = make_optimizer(learning_rate=3e-4)
    critic_optimizer = make_optimizer(learning_rate=3e-4)
    alpha_optimizer = make_optimizer(learning_rate=1e-3)
    return (
        buffer,
        actor,
        critic,
        alpha_network,
        actor_optimizer,
        critic_optimizer,
        alpha_optimizer,
    )


@pytest.fixture()
def pqn_components(dummy_env_params: DummyEnvParams):
    q_network = make_discrete_network(
        obs_dim=dummy_env_params.obs_dim, action_dim=dummy_env_params.action_dim
    )
    optimizer = make_optimizer(learning_rate=3e-4)
    schedule = make_linear_schedule(start=1.0, end=0.01, steps=100)
    return q_network, optimizer, schedule
