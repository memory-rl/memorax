import chex
import jax
import jax.numpy as jnp
from optax import incremental_update, periodic_update
from optax._src import base


def periodic_incremental_update(
    new_tensors: base.Params,
    old_tensors: base.Params,
    steps: chex.Array,
    update_period: int,
    step_size: chex.Numeric,
) -> base.Params:
    """Periodically perform Polyak-style incremental updates.

    Combines the ideas of `periodic_update` and `incremental_update`: every
    `update_period` steps, the slow copy is updated with an exponential moving
    average of the fast parameters; otherwise it stays unchanged.

    Args:
      new_tensors: the latest value of the tensors.
      old_tensors: a slow copy of the model's parameters.
      steps: current number of update steps on the "online" network.
      update_period: every how many steps to refresh the slow copy.
      step_size: Polyak averaging factor used when the refresh occurs.

    Returns:
      a slow copy of the model's parameters that is incrementally updated every
      `update_period` steps:
      `step_size * new_tensors + (1 - step_size) * old_tensors`.
    """
    return periodic_update(
        incremental_update(new_tensors, old_tensors, step_size),
        old_tensors,
        steps,
        update_period,
    )


def delayed_update(
    new_tensors: base.Params,
    old_tensors: base.Params,
    new_opt_state: base.OptState,
    old_opt_state: base.OptState,
    steps: chex.Array,
    start_step: int,
) -> base.Params:
    """Update all parameters only after a given timestep is reached.

    Args:
      new_tensors: the latest value of the tensors.
      old_tensors: a copy of the model's parameters that remains unchanged
        until `steps >= start_step`.
      steps: current number of update steps on the "online" network.
      start_step: timestep at which the copy begins mirroring `new_tensors`.

    Returns:
      a copy of the model's parameters that equals `old_tensors` before
      `start_step` and `new_tensors` from `start_step` onward.
    """
    return jax.lax.cond(
        steps >= start_step,
        lambda _: new_tensors,
        new_opt_state,
        lambda _: old_tensors,
        old_opt_state,
        None,
    )
