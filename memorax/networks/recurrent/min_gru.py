from functools import partial

import jax
import jax.numpy as jnp
from jax import lax
from flax import linen as nn
from flax.linen import initializers
from flax.linen.module import compact
from typing import Any, Callable, Tuple, Optional

# Type aliases
PRNGKey = Any
Shape = Tuple[int, ...]
Dtype = Any
Array = Any
Initializer = Callable[[PRNGKey, Shape, Dtype], Array]


class MinGRU(nn.Module):
    features: int
    gate_fn: Callable[..., Any] = nn.sigmoid
    activation_fn: Callable[..., Any] = nn.tanh
    kernel_init: Initializer = initializers.lecun_normal()
    bias_init: Initializer = initializers.zeros_init()
    dtype: Optional[Dtype] = None
    param_dtype: Dtype = jnp.float32

    @compact
    def __call__(
        self, inputs: Array, mask: Array, initial_carry: Optional[Array] = None
    ) -> Array:

        dense_i = partial(
            nn.Dense,
            features=self.features,
            use_bias=True,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
        )

        z = self.gate_fn(dense_i(name="iz")(inputs))

        n = self.activation_fn(dense_i(name="in")(inputs))

        a = z * (1 - mask[..., None])
        b = (1.0 - z) * n

        def scan_fn(prev_values, values):
            a_prev, b_prev = prev_values
            a, b = values

            a_new = a * a_prev
            b_new = a * b_prev + b
            return a_new, b_new

        if initial_carry is not None:
            a, b = lax.associative_scan(jax.vmap(scan_fn), (a, b), axis=1)

            carry = a * initial_carry[:, None, :] + b
        else:
            _, carry = lax.associative_scan(jax.vmap(scan_fn), (a, b), axis=1)

        return carry, carry[:, -1, :]

    def initialize_carry(
        self, rng: Optional[PRNGKey], input_shape: Tuple[int, ...]
    ) -> Array:
        batch_dims = input_shape[:-1]
        mem_shape = batch_dims + (self.features,)

        return jnp.zeros(mem_shape, dtype=self.dtype or self.param_dtype)

    @property
    def num_feature_axes(self) -> int:
        return 1
