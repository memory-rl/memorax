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

    Uses kernel feature maps (ELU+1) to linearize attention, enabling
    efficient parallel computation via associative scan.

    Based on "Transformers are RNNs" (Katharopoulos et al., 2020).

    Element: (state, normalizer) where:
        - state: outer product Σ φ(k) ⊗ v
        - normalizer: sum of keys Σ φ(k)
    Combine: element-wise addition of states and normalizers
    """

    head_dim: int
    num_heads: int
    kernel_init: Initializer
    bias_init: Initializer
    param_dtype: Dtype
    dtype: Optional[Dtype] = None
    eps: float = 1e-6

    def _feature_map(self, x: Array) -> Array:
        """ELU+1 feature map as in the original paper."""
        return nn.elu(x) + 1

    @nn.compact
    def __call__(self, x: Array, **kwargs) -> Carry:
        """Compute key-value outer products for memory storage.

        Args:
            x: Input of shape (B, T, D)

        Returns:
            Carry tuple of (state, normalizer) where:
                - state: outer product (B, T, H, head_dim, head_dim)
                - normalizer: sum of keys (B, T, H, head_dim)
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

        # Apply feature map to keys
        key = self._feature_map(key)

        # State: outer product v ⊗ φ(k)
        state = jnp.einsum("bthi,bthj->bthij", value, key)

        # Normalizer: sum of φ(k) for proper normalization
        normalizer = key

        return (state, normalizer)

    def combine(self, a: Carry, b: Carry) -> Carry:
        """Combine two elements via addition."""
        state_i, norm_i = a
        state_j, norm_j = b
        return (state_i + state_j, norm_i + norm_j)

    @nn.compact
    def read(self, h: Carry, x: Array, **kwargs) -> Array:
        """Query accumulated memory to produce output.

        Args:
            h: Accumulated state (state, normalizer)
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
        query = self._feature_map(query)

        state, normalizer = h

        # Numerator: φ(q) @ S = φ(q) @ (Σ v ⊗ φ(k))
        numerator = jnp.einsum("bthij,bthj->bthi", state, query)

        # Denominator: φ(q) @ z = φ(q) @ (Σ φ(k))
        denominator = jnp.einsum("bthi,bthi->bth", query, normalizer)
        denominator = jnp.maximum(denominator, self.eps)[:, :, :, None]

        # Normalized output
        output = numerator / denominator

        hidden_dim = self.num_heads * self.head_dim
        output = output.reshape(batch_size, sequence_length, hidden_dim)

        output = nn.RMSNorm(dtype=self.dtype)(output)
        output = nn.Dense(
            features=in_features,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
        )(output)

        return output

    def initialize_carry(self, key: jax.Array, input_shape: Tuple[int, ...]) -> Carry:
        """Initialize carry with zero state and normalizer."""
        batch_size, *_ = input_shape
        state = jnp.zeros(
            (batch_size, 1, self.num_heads, self.head_dim, self.head_dim)
        )
        normalizer = jnp.zeros((batch_size, 1, self.num_heads, self.head_dim))
        return (state, normalizer)
