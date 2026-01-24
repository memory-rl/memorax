from functools import partial
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.typing import Dtype, Initializer

from memorax.utils.typing import Array, Carry

from .memoroid import Algebra


class LinearAttention(Algebra):
    """Linear attention as a memoroid algebra.

    Uses kernel feature maps (ReLU) to linearize attention, enabling
    efficient parallel computation via associative scan.

    Element: (decay, state) where state is the outer product of value and key
    Combine: (decay_j * decay_i, decay_j * state_i + state_j)
    """

    head_dim: int
    num_heads: int
    kernel_init: Initializer
    bias_init: Initializer
    param_dtype: Dtype
    dtype: Optional[Dtype] = None

    @nn.compact
    def __call__(self, x: Array, **kwargs) -> Carry:
        """Compute key-value outer products for memory storage.

        Args:
            x: Input of shape (B, T, D)

        Returns:
            Carry tuple of (decay, state) where:
                - decay: scalar 1.0 broadcast to (B, T, H, 1, 1)
                - state: outer product (B, T, H, head_dim, head_dim)
        """
        batch_size, sequence_length, _ = x.shape

        projection = partial(
            nn.DenseGeneral,
            features=(self.num_heads, self.head_dim),
            use_bias=False,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
        )

        key = projection(name="key")(x)
        value = projection(name="value")(x)

        key = nn.relu(key)

        state = jnp.einsum("bthi,bthj->bthij", value, key)

        decay = jnp.ones((batch_size, sequence_length, self.num_heads, 1, 1))

        return (decay, state)

    def combine(self, a: Carry, b: Carry) -> Carry:
        """Combine two elements via decay-weighted accumulation."""
        decay_i, state_i = a
        decay_j, state_j = b
        return (decay_j * decay_i, decay_j * state_i + state_j)

    @nn.compact
    def read(self, h: Carry, x: Array, **kwargs) -> Array:
        """Query accumulated memory to produce output.

        Args:
            h: Accumulated state (decay, state)
            x: Original input of shape (B, T, D)

        Returns:
            Output of shape (B, T, D)
        """
        batch_size, sequence_length, in_features = x.shape

        projection = partial(
            nn.DenseGeneral,
            features=(self.num_heads, self.head_dim),
            use_bias=False,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
        )

        query = projection(name="query")(x)
        query = nn.relu(query)

        _, state = h

        hidden_dim = self.num_heads * self.head_dim
        output = jnp.einsum("bthij,bthj->bthi", state, query).reshape(
            batch_size, sequence_length, hidden_dim
        )

        output = nn.RMSNorm(dtype=self.dtype)(output)
        output = nn.Dense(
            features=in_features,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
        )(output)

        return output

    def initialize_carry(self, key: jax.Array, input_shape: Tuple[int, ...]) -> Carry:
        """Initialize carry with identity decay and zero state."""
        batch_size, *_ = input_shape
        decay = jnp.ones((batch_size, 1, self.num_heads, 1, 1))
        state = jnp.zeros(
            (batch_size, 1, self.num_heads, self.head_dim, self.head_dim)
        )
        return (decay, state)
