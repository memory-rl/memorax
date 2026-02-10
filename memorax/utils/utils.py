from typing import Optional

import jax.numpy as jnp


def broadcast(x: Optional[jnp.ndarray], to: jnp.ndarray):
    if x is None:
        return x
    if x.ndim > to.ndim:
        raise ValueError(
            f"Cannot broadcast array with ndim={x.ndim} to target with ndim={to.ndim}. "
            "Source has more dimensions than target."
        )
    while x.ndim < to.ndim:
        x = x[..., None]
    return x
