import jax
import jax.numpy as jnp


@jax.jit
def generalized_advantage_estimatation(gamma: float, gae_lambda: float, final_value: jax.Array, transitions):
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
        unroll=16,
    )
    returns = advantages + transitions.value
    return advantages, returns
