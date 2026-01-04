from typing import Optional

import jax.numpy as jnp

def broadcast(x: Optional[jnp.ndarray], to: jnp.ndarray):
    if x is None:
        return x
    while x.ndim != to.ndim:
        x = x[..., None] if x.ndim < to.ndim else x[..., 0]
    return x
