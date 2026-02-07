import jax
import jax.numpy as jnp


@jax.jit
def vtrace(
    gamma: float,
    gae_lambda: float,
    final_value: jax.Array,
    transitions,
    importance_ratios: jax.Array,
    rho_clip: float = 1.0,
    c_clip: float = 1.0,
):
    """Compute V-trace targets and advantages for a trajectory."""

    def f(carry, inputs):
        advantage, value = carry
        transition, importance_ratio = inputs

        rho = jnp.minimum(rho_clip, importance_ratio)
        c = jnp.minimum(c_clip, importance_ratio)

        delta = rho * (
            transition.reward + gamma * value * (1 - transition.done) - transition.value
        )
        advantage = delta + gamma * gae_lambda * (1 - transition.done) * c * advantage
        return (advantage, transition.value), advantage

    _, advantages = jax.lax.scan(
        f,
        (jnp.zeros_like(final_value), final_value),
        (transitions, importance_ratios),
        reverse=True,
        unroll=16,
    )
    returns = advantages + transitions.value
    return advantages, returns
