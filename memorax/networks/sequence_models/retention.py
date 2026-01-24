from functools import partial
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.typing import Dtype, Initializer

from memorax.utils.typing import Array, Carry

from .memoroid import MemoroidCellBase


class RetentionCell(MemoroidCellBase):
    """Multi-scale retention as a memoroid algebra.

    Implements the retention mechanism from "Retentive Network: A Successor
    to Transformer for Large Language Models" (Sun et al., 2023).

    Retention uses exponential decay instead of softmax attention:
        S_t = γ * S_{t-1} + k_t ⊗ v_t
        o_t = q_t @ S_t

    Multi-scale retention uses different decay rates (γ) per head, enabling
    the model to capture patterns at different temporal scales.

    Element: (state, decay) where:
        - state: outer product k ⊗ v
        - decay: per-head decay factor γ
    Combine: (state_a * decay_b + state_b, decay_a * decay_b)
    """

    head_dim: int
    num_heads: int
    kernel_init: Initializer
    bias_init: Initializer
    param_dtype: Dtype
    dtype: Optional[Dtype] = None
    gamma_min: float = 0.9
    gamma_max: float = 0.99

    @property
    def gammas(self) -> Array:
        """Multi-scale decay: linearly spaced γ values per head.

        Head 0 gets gamma_max (long-term), head N-1 gets gamma_min (short-term).
        """
        return jnp.linspace(self.gamma_max, self.gamma_min, self.num_heads)

    @nn.compact
    def __call__(self, x: Array, **kwargs) -> Carry:
        """Compute key-value outer products with decay factors.

        Args:
            x: Input of shape (B, T, D)

        Returns:
            Carry tuple of (state, decay) where:
                - state: outer product (B, T, H, head_dim, head_dim)
                - decay: broadcast decay (B, T, H, 1, 1)
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

        # State: outer product k ⊗ v
        state = jnp.einsum("bthk,bthv->bthkv", key, value)

        # Decay factor broadcast to (B, T, H, 1, 1) for element-wise ops
        decay = self.gammas[None, None, :, None, None]
        decay = jnp.broadcast_to(decay, (batch_size, sequence_length, self.num_heads, 1, 1))

        return (state, decay)

    def binary_operator(self, a: Carry, b: Carry) -> Carry:
        """Combine two elements with exponential decay.

        The key insight: state_a should be decayed by decay_b before adding state_b.
        This makes the operation associative while implementing recurrent decay.
        """
        state_a, decay_a = a
        state_b, decay_b = b

        # Decay state_a by decay_b, then add state_b
        state_combined = state_a * decay_b + state_b

        # Accumulate decay factors
        decay_combined = decay_a * decay_b

        return (state_combined, decay_combined)

    @nn.compact
    def read(self, h: Carry, x: Array, **kwargs) -> Array:
        """Query accumulated memory to produce output.

        Args:
            h: Accumulated state (state, decay)
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

        state, _ = h

        # Output: q @ S (query contracts with state)
        output = jnp.einsum("bthk,bthkv->bthv", query, state)

        # Reshape and project
        hidden_dim = self.num_heads * self.head_dim
        output = output.reshape(batch_size, sequence_length, hidden_dim)

        # Group normalization (as in RetNet paper)
        output = nn.GroupNorm(num_groups=self.num_heads, dtype=self.dtype)(output)

        # Output projection
        output = nn.Dense(
            features=in_features,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
        )(output)

        return output

    def initialize_carry(self, key: jax.Array, input_shape: Tuple[int, ...]) -> Carry:
        """Initialize carry with zero state and unit decay."""
        batch_size, *_ = input_shape
        state = jnp.zeros(
            (batch_size, 1, self.num_heads, self.head_dim, self.head_dim),
            dtype=self.dtype,
        )
        # Initial decay is 1.0 (no accumulated decay yet)
        decay = jnp.ones(
            (batch_size, 1, self.num_heads, 1, 1),
            dtype=self.dtype or jnp.float32,
        )
        return (state, decay)
