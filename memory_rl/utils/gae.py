import jax
import jax.numpy as jnp


@jax.jit
def compute_gae(gamma: float, gae_lambda: float, final_value: jax.Array, transitions):
    """Compute Generalized Advantage Estimates (GAE) for a trajectory."""

    def f(carry, transition):
        advantage, value = carry
        delta = (
            transition.reward + gamma * value * (1 - transition.done) - transition.value
        )
        advantage = delta + gamma * gae_lambda * (1 - transition.done) * advantage
        return (advantage, transition.value), advantage

    _, advantages = jax.lax.scan(
        f,
        (jnp.zeros_like(final_value), final_value),
        transitions,
        reverse=True,
    )
    returns = advantages + transitions.value
    return advantages, returns


@jax.jit
def compute_recurrent_gae(
    gamma: float,
    gae_lambda: float,
    transitions,
    final_value: jax.Array,
    final_done,
):
    """Compute Generalized Advantage Estimates (GAE) for a trajectory."""

    def f(carry, transition):
        advantage, next_value, next_done = carry
        delta = (
            transition.reward + gamma * next_value * (1 - next_done) - transition.value
        )
        advantage = delta + gamma * gae_lambda * (1 - next_done) * advantage
        return (advantage, transition.value, transition.done), advantage

    _, advantages = jax.lax.scan(
        f,
        (jnp.zeros_like(final_value), final_value, final_done),
        transitions,
        reverse=True,
    )
    returns = advantages + transitions.value
    return advantages, returns
