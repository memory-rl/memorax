import gymnax
import jax
import pytest


@pytest.fixture
def cartpole_env():
    """Returns (env, env_params) for CartPole-v1."""
    env, env_params = gymnax.make("CartPole-v1")
    return env, env_params


@pytest.fixture
def pendulum_env():
    """Returns (env, env_params) for Pendulum-v1."""
    env, env_params = gymnax.make("Pendulum-v1")
    return env, env_params


@pytest.fixture
def random_key():
    """Returns a JAX PRNGKey."""
    return jax.random.key(42)
