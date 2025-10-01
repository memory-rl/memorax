from functools import partial
from typing import (
    Any,
    TypeVar,
)

import jax
from jax import numpy as jnp
from jax import random

from flax.linen import LayerNorm, initializers
from flax.linen.activation import sigmoid
from flax.linen.linear import Dense
from flax.linen.module import compact, nowrap
from flax.typing import (
    Array,
    PRNGKey,
    Dtype,
    Initializer,
)
from flax.linen.recurrent import RNNCellBase

A = TypeVar("A")
Carry = Any
CarryHistory = Any
Output = Any


class SHMCell(RNNCellBase):
    r"""Stable Hadamard Memory (SHM) cell."""

    features: int
    output_features: int | None = None
    num_thetas: int = 128
    sample_theta: bool = True

    kernel_init: Initializer = initializers.variance_scaling(
        2.0, "fan_in", "uniform"
    )  # Kaiming uniform
    bias_init: Initializer = initializers.zeros_init()
    theta_init: Initializer = initializers.variance_scaling(
        1.0, "fan_avg", "uniform"
    )  # Xavier uniform
    dtype: Dtype | None = None
    param_dtype: Dtype = jnp.float32
    carry_init: Initializer = initializers.zeros_init()

    @compact
    def __call__(self, carry: Array, inputs: Array) -> tuple[Array, Array]:

        dense = partial(
            Dense,
            use_bias=False,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
        )

        inputs = LayerNorm(
            epsilon=1e-5, dtype=self.dtype, param_dtype=self.param_dtype, name="ln"
        )(inputs)

        v = dense(name="v", features=self.features)(inputs)
        k = jax.nn.relu(dense(name="k", features=self.features)(inputs))
        q = jax.nn.relu(dense(name="q", features=self.features)(inputs))
        v_c = dense(name="vc", features=self.features)(inputs)
        eta = sigmoid(dense(name="eta", features=1)(inputs))

        k = k / (1e-5 + jnp.sum(k, axis=-1, keepdims=True))
        q = q / (1e-5 + jnp.sum(q, axis=-1, keepdims=True))

        U = ((eta * v)[..., :, None]) * k[..., None, :]

        theta_table = self.param(
            "theta_table",
            self.theta_init,
            (self.num_thetas, self.features),
            self.param_dtype,
        )

        if self.sample_theta and self.has_rng("memory"):
            rng = self.make_rng("memory")
            batch_shape = v_c.shape[:-1]
            idx = random.randint(rng, batch_shape, 0, self.num_thetas, dtype=jnp.int32)
            theta_t = theta_table[idx]
            theta_t = jnp.broadcast_to(theta_t, v_c.shape)
        else:
            theta_t = jnp.broadcast_to(theta_table[0], v_c.shape)

        C = 1.0 + jnp.tanh(theta_t[..., :, None] * v_c[..., None, :])

        carry = carry * C + U

        h = jnp.einsum("...ij,...j->...i", carry, q)

        if self.output_features is not None:
            h = Dense(
                features=self.output_features,
                use_bias=True,
                dtype=self.dtype,
                kernel_init=self.kernel_init,
                bias_init=self.bias_init,
                name="out",
            )(h)

        return carry, h

    @nowrap
    def initialize_carry(self, rng: PRNGKey, input_shape: tuple[int, ...]) -> Array:
        batch_dims = input_shape[:-1]
        mem_shape = batch_dims + (self.features, self.features)
        return self.carry_init(rng, mem_shape, self.param_dtype)

    @property
    def num_feature_axes(self) -> int:
        return 1
