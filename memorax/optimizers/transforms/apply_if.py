from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import optax


class ApplyIfState(NamedTuple):
    total_skipped: jnp.ndarray
    inner_state: Any


def apply_if(
    inner: optax.GradientTransformation,
) -> optax.GradientTransformationExtraArgs:
    inner = optax.with_extra_args_support(inner)

    def init(params):
        return ApplyIfState(
            inner_state=inner.init(params), total_skipped=jnp.array(0, dtype=jnp.int32)
        )

    def update(updates, state, params=None, **extra_args):
        skip = extra_args.get("skip", False)

        def do_update(_):
            new_updates, new_inner_state = inner.update(
                updates, state.inner_state, params, **extra_args
            )
            return new_updates, new_inner_state, state.total_skipped

        def reject_update(_):
            return (
                optax.tree_utils.tree_zeros_like(updates),
                state.inner_state,
                state.total_skipped + 1,
            )

        updates, new_inner_state, new_total_skipped = jax.lax.cond(
            skip, reject_update, do_update, operand=None
        )

        return updates, ApplyIfState(
            inner_state=new_inner_state, total_skipped=new_total_skipped
        )

    return optax.GradientTransformationExtraArgs(init=init, update=update)
