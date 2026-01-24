from functools import partial
from typing import Any, TypeVar

import flax.linen as nn
import jax
from flax.linen import LayerNorm, initializers
from flax.linen.activation import sigmoid
from flax.linen.linear import Dense, default_kernel_init
from flax.linen.module import compact, nowrap
from flax.linen.recurrent import RNNCellBase
from flax.typing import Array, Dtype, Initializer, PRNGKey
from jax import numpy as jnp
from jax import random
from memorax.networks.sequence_models.utils import xavier_uniform

A = TypeVar("A")
Carry = Any
CarryHistory = Any
Output = Any


class SHMCell(RNNCellBase):
    """Stable Hadamard Memory (SHM) cell."""

    features: int
    output_features: int
    num_thetas: int = 128
    sample_theta: bool = True

    kernel_init: Initializer = default_kernel_init
    bias_init: Initializer = initializers.zeros_init()
    theta_init: Initializer = xavier_uniform()
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

        v = dense(name="value", features=self.features)(inputs)
        k = jax.nn.relu(dense(name="key", features=self.features)(inputs))
        q = jax.nn.relu(dense(name="query", features=self.features)(inputs))
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

        y = nn.Dense(self.output_features, name="output")(h)

        return carry, y

    @nowrap
    def initialize_carry(self, rng: PRNGKey, input_shape: tuple[int, ...]) -> Array:
        batch_dims = input_shape[:-1]
        mem_shape = batch_dims + (self.features, self.features)
        return self.carry_init(rng, mem_shape, self.param_dtype)

    @property
    def num_feature_axes(self) -> int:
        return 1
