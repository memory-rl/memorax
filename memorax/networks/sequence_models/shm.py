from functools import partial
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.linen import initializers
from flax.linen.recurrent import RNNCellBase
from flax.typing import Dtype, Initializer
from jax import random

from memorax.utils.typing import Array, Carry

from .utils import xavier_uniform


class SHMCell(RNNCellBase):
    """Stable Hadamard Memory (SHM) cell."""

    features: int
    output_features: int
    num_thetas: int = 128
    sample_theta: bool = True

    kernel_init: Initializer = nn.initializers.lecun_normal()
    bias_init: Initializer = initializers.zeros_init()
    theta_init: Initializer = xavier_uniform()
    dtype: Optional[Dtype] = None
    param_dtype: Dtype = jnp.float32
    carry_init: Initializer = initializers.zeros_init()

    def setup(self):
        dense = partial(
            nn.Dense,
            use_bias=False,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
        )

        self.ln = nn.LayerNorm(
            epsilon=1e-5, dtype=self.dtype, param_dtype=self.param_dtype,
        )
        self.value = dense(features=self.features)
        self.key = dense(features=self.features)
        self.query = dense(features=self.features)
        self.v_c = dense(features=self.features)
        self.eta = dense(features=1)
        self.theta_table = self.param(
            "theta_table",
            self.theta_init,
            (self.num_thetas, self.features),
            self.param_dtype,
        )
        self.output_projection = nn.Dense(self.output_features)

    def __call__(self, carry: Array, inputs: Array) -> Tuple[Array, Array]:
        inputs = self.ln(inputs)

        value = self.value(inputs)
        key = jax.nn.relu(self.key(inputs))
        query = jax.nn.relu(self.query(inputs))
        v_c = self.v_c(inputs)
        eta_val = nn.sigmoid(self.eta(inputs))

        key = key / (1e-5 + jnp.sum(key, axis=-1, keepdims=True))
        query = query / (1e-5 + jnp.sum(query, axis=-1, keepdims=True))

        U = ((eta_val * value)[..., :, None]) * key[..., None, :]

        if self.sample_theta and self.has_rng("memory"):
            rng = self.make_rng("memory")
            batch_shape = v_c.shape[:-1]
            idx = random.randint(rng, batch_shape, 0, self.num_thetas, dtype=jnp.int32)
            theta_t = self.theta_table[idx]
            theta_t = jnp.broadcast_to(theta_t, v_c.shape)
        else:
            theta_t = jnp.broadcast_to(self.theta_table[0], v_c.shape)

        C = 1.0 + jnp.tanh(theta_t[..., :, None] * v_c[..., None, :])

        M = carry * C + U

        h = jnp.einsum("...ij,...j->...i", M, query)

        y = self.output_projection(h)

        return M, y

    @nn.nowrap
    def initialize_carry(self, key: jax.Array, input_shape: Tuple[int, ...]) -> Array:
        batch_dims = input_shape[:-1]
        mem_shape = batch_dims + (self.features, self.features)
        return self.carry_init(key, mem_shape, self.param_dtype)

    @property
    def num_feature_axes(self) -> int:
        return 1
