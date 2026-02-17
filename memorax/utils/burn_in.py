import jax

from memorax.networks.sequence_models.utils import add_feature_axis


def burn_in(network, params, carry, burn_in_transitions, *, target=False):
    """Warm up carry by running a detached forward pass over burn-in transitions."""
    if target:
        obs, mask, action, reward, done = (
            burn_in_transitions.next_obs,
            burn_in_transitions.done,
            burn_in_transitions.action,
            burn_in_transitions.reward,
            burn_in_transitions.done,
        )
    else:
        obs, mask, action, reward, done = (
            burn_in_transitions.obs,
            burn_in_transitions.prev_done,
            burn_in_transitions.prev_action,
            burn_in_transitions.prev_reward,
            burn_in_transitions.prev_done,
        )
    carry, _ = network.apply(
        jax.lax.stop_gradient(params),
        observation=obs,
        mask=mask,
        action=action,
        reward=add_feature_axis(reward),
        done=done,
        initial_carry=carry,
    )
    return jax.lax.stop_gradient(carry)
