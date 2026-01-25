from functools import partial
from typing import Any, Callable, Optional, Tuple

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.linen import initializers

from memorax.utils.typing import Array, Carry

from .memoroid import MemoroidCellBase

PRNGKey = Any
Shape = Tuple[int, ...]
Dtype = Any
Initializer = Callable[[PRNGKey, Shape, Dtype], Array]


class MinGRUCell(MemoroidCellBase):
    """Minimal GRU algebra using log-space computation.

    Operates in log-space to avoid numerical overflow for long sequences.

    Element: (log_state, cumulative_decay)
    Combine: (logaddexp(decay_j + log_state_i, log_state_j), decay_i + decay_j)
    """

    features: int
    kernel_init: Initializer = initializers.lecun_normal()
    bias_init: Initializer = initializers.zeros_init()
    dtype: Optional[Dtype] = None
    param_dtype: Dtype = jnp.float32

    @nn.compact
    def __call__(self, inputs: Array, **kwargs) -> Carry:
        B, T, _ = inputs.shape

        dense = partial(
            nn.Dense,
            features=self.features,
            use_bias=False,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
        )

        z = dense(name="z")(inputs)
        h_tilde = dense(name="h")(inputs)

        # Compute in log-space
        log_z = -nn.softplus(-z)
        log_h_tilde = jnp.where(
            h_tilde >= 0, jnp.log(nn.relu(h_tilde) + 0.5), -nn.softplus(-h_tilde)
        )

        log_state = log_z + log_h_tilde
        decay = -nn.softplus(z)

        return (log_state, decay)

    def binary_operator(self, a: Carry, b: Carry) -> Carry:
        """Log-space combine: logaddexp for numerically stable addition."""
        log_state_i, decay_i = a
        log_state_j, decay_j = b
        return (
            jnp.logaddexp(decay_j + log_state_i, log_state_j),
            decay_i + decay_j,
        )

    def read(self, h: Carry, x: Array, **kwargs) -> Array:
        log_state, _ = h
        return jnp.exp(log_state)

    def initialize_carry(self, key: jax.Array, input_shape: Tuple[int, ...]) -> Carry:
        *batch_dims, _ = input_shape
        # Identity for logaddexp is -inf, identity for + is 0
        log_state = jnp.full(
            (*batch_dims, 1, self.features),
            -jnp.inf,
            dtype=self.dtype or self.param_dtype,
        )
        decay = jnp.zeros(
            (*batch_dims, 1, self.features), dtype=self.dtype or self.param_dtype
        )
        return (log_state, decay)
