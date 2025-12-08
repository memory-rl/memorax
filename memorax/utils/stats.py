import jax.numpy as jnp


def naniqm(x, axis=None, keepdims=False):
    q1 = jnp.nanquantile(x, 0.25, axis=axis, keepdims=True)
    q3 = jnp.nanquantile(x, 0.75, axis=axis, keepdims=True)
    mask = (x >= q1) & (x <= q3) & ~jnp.isnan(x)
    nominator = jnp.where(mask, x, 0).sum(
        axis=axis, keepdims=keepdims or (axis is None)
    )
    denominator = mask.sum(axis=axis, keepdims=keepdims or (axis is None))
    out = nominator / jnp.maximum(denominator, 1)
    return out if keepdims or axis is None else jnp.squeeze(out, axis=axis)
