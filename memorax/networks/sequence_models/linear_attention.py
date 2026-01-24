from functools import partial
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.linen import compact
from flax.typing import Dtype, Initializer
from jax import lax
from memorax.networks.sequence_models.sequence_model import SequenceModel
from memorax.networks.sequence_models.utils import broadcast_mask
from memorax.utils.typing import Array, Carry


class LinearAttention(SequenceModel):
    head_dim: int
    num_heads: int
    kernel_init: Initializer
    bias_init: Initializer
    param_dtype: Dtype
    dtype: Optional[Dtype]

    def initialize_carry(self, key, input_shape) -> Carry:
        batch_size, *_ = input_shape
        return jnp.zeros((batch_size, self.num_heads, self.head_dim, self.head_dim))

    @compact
    def __call__(
        self,
        inputs: Array,
        mask: Array,
        initial_carry: Optional[Carry] = None,
        **kwargs,
    ) -> Tuple[Array, Array]:
        batch_size, sequence_length, in_features = inputs.shape

        if initial_carry is None:
            initial_carry = jnp.zeros(
                (batch_size, self.num_heads, self.head_dim, self.head_dim)
            )

        projection = partial(
            nn.DenseGeneral,
            features=(self.num_heads, self.head_dim),
            use_bias=False,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
        )

        query = projection(name="query")(inputs)
        key = projection(name="key")(inputs)
        value = projection(name="value")(inputs)

        query = nn.relu(query)
        key = nn.relu(key)

        outer_product = jnp.einsum("bshi,bshj->bshij", value, key)

        decay = jnp.broadcast_to(jnp.eye(self.head_dim), outer_product.shape)
        decay = decay * (1.0 - broadcast_mask(mask, decay))

        def binary_operator(lhs, rhs):
            decay_i, outer_i = lhs
            decay_j, outer_j = rhs
            return (
                jnp.einsum("bhij,bhjk->bhik", decay_j, decay_i),
                jnp.einsum("bhij,bhjk->bhik", decay_j, outer_i) + outer_j,
            )

        cumulative_decay, state = lax.associative_scan(
            binary_operator, (decay, outer_product), axis=1
        )

        carry = jnp.einsum("bshij,bhjk->bshik", cumulative_decay, initial_carry) + state

        hidden_dim = self.num_heads * self.head_dim
        x = jnp.einsum("bshij,bshj->bshi", carry, query).reshape(
            batch_size, sequence_length, hidden_dim
        )

        x = nn.RMSNorm(dtype=self.dtype)(x)

        y = nn.Dense(
            features=in_features, kernel_init=self.kernel_init, bias_init=self.bias_init
        )(x)

        final_carry = carry[:, -1]

        return final_carry, y
