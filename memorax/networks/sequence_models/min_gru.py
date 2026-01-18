from functools import partial
from typing import Any, Callable, Optional, Tuple

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.linen import initializers
from flax.linen.module import compact
from jax import lax

from .sequence_model import SequenceModel

PRNGKey = Any
Shape = Tuple[int, ...]
Dtype = Any
Array = Any
Initializer = Callable[[PRNGKey, Shape, Dtype], Array]


class MinGRU(SequenceModel):
    features: int
    kernel_init: Initializer = initializers.lecun_normal()
    bias_init: Initializer = initializers.zeros_init()
    dtype: Optional[Dtype] = None
    param_dtype: Dtype = jnp.float32

    @compact
    def __call__(
        self,
        inputs: Array,
        mask: Array,
        initial_carry: Optional[Array] = None,
        **kwargs,
    ) -> Array:
        _, sequence_length, _ = inputs.shape
        mask = mask[..., None]

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

        if sequence_length == 1:
            z = nn.sigmoid(z)
            h_tilde = jnp.where(h_tilde >= 0, h_tilde + 0.5, nn.sigmoid(h_tilde))

            if initial_carry is not None:
                initial_carry = jnp.where(mask, 0.0, initial_carry)
                carry = y = initial_carry + z * (h_tilde - initial_carry)
            else:
                carry = y = z * h_tilde

            return carry, y

        log_z = -nn.softplus(-z)
        log_h_tilde = jnp.where(
            h_tilde >= 0, jnp.log(nn.relu(h_tilde) + 0.5), -nn.softplus(-h_tilde)
        )

        x = log_z + log_h_tilde
        decay = jnp.where(mask, -jnp.inf, -nn.softplus(z))

        if initial_carry is not None:
            x = jnp.concatenate([jnp.log(initial_carry), x], axis=1)
            decay = jnp.pad(decay, ((0, 0), (1, 0), (0, 0)), constant_values=0.0)

        def binary_operation(lhs, rhs):
            x_i, decay_i = lhs
            x_j, decay_j = rhs
            return jnp.logaddexp(decay_j + x_i, x_j), decay_i + decay_j

        log_h, _ = jax.lax.associative_scan(binary_operation, (x, decay), axis=1)
        h = jnp.exp(log_h)

        if initial_carry is not None:
            h = h[:, 1:, :]

        carry = h[:, -1:, :]
        y = h[:, -sequence_length:, :]

        return carry, y

    def initialize_carry(
        self, rng: Optional[PRNGKey], input_shape: Tuple[int, ...]
    ) -> Array:
        batch_size, *_ = input_shape
        mem_shape = (
            batch_size,
            1,
            self.features,
        )

        return jnp.zeros(mem_shape, dtype=self.dtype or self.param_dtype)

    @property
    def num_feature_axes(self) -> int:
        return 1
