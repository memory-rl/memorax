from functools import partial
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.typing import Dtype, Initializer

from memorax.utils.typing import Array, Carry

from .memoroid import MemoroidCellBase


class LinearAttentionCell(MemoroidCellBase):
    """Linear attention as a memoroid algebra.

    Uses kernel feature maps (ELU+1) to linearize attention, enabling
    efficient parallel computation via associative scan.

    Based on "Transformers are RNNs" (Katharopoulos et al., 2020).

    Element: (S, z) where:
        - S: outer product Σ φ(k) ⊗ v
        - z: sum of keys Σ φ(k)
    Combine: element-wise addition of S and z
    """

    features: int
    head_dim: int
    num_heads: int
    kernel_init: Initializer = nn.initializers.lecun_normal()
    bias_init: Initializer = nn.initializers.zeros_init()
    dtype: Optional[Dtype] = None
    param_dtype: Dtype = jnp.float32
    eps: float = 1e-6

    def setup(self):
        projection = partial(
            nn.DenseGeneral,
            features=(self.num_heads, self.head_dim),
            use_bias=False,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
        )

        self.key = projection(name="key")
        self.value = projection(name="value")
        self.query = projection(name="query")
        self.norm = nn.RMSNorm(dtype=self.dtype)
        self.output_projection = nn.Dense(
            features=self.features,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
        )

    def _feature_map(self, x: Array) -> Array:
        """ELU+1 feature map as in the original paper."""
        return nn.elu(x) + 1

    def __call__(self, x: Array, **kwargs) -> Carry:
        """Compute key-value outer products for memory storage.

        Args:
            x: Input of shape (B, T, D)

        Returns:
            Carry tuple of (S, z) where:
                - S: outer product (B, T, H, head_dim, head_dim)
                - z: sum of keys (B, T, H, head_dim)
        """
        B, T, _ = x.shape

        key = self._feature_map(self.key(x))
        value = self.value(x)

        S = jnp.einsum("bthi,bthj->bthij", value, key)

        return (S, key)

    def binary_operator(self, a: Carry, b: Carry) -> Carry:
        """Combine two elements via addition."""
        S_i, z_i = a
        S_j, z_j = b
        return (S_i + S_j, z_i + z_j)

    def read(self, h: Carry, x: Array, **kwargs) -> Array:
        """Query accumulated memory to produce output.

        Args:
            h: Accumulated state (S, z)
            x: Original input of shape (B, T, D)

        Returns:
            Output of shape (B, T, D)
        """
        B, T, _ = x.shape

        query = self._feature_map(self.query(x))

        S, z = h

        numerator = jnp.einsum("bthij,bthj->bthi", S, query)

        denominator = jnp.einsum("bthi,bthi->bth", query, z)
        denominator = jnp.maximum(denominator, self.eps)[:, :, :, None]

        output = numerator / denominator

        hidden_dim = self.num_heads * self.head_dim
        output = output.reshape(B, T, hidden_dim)

        output = self.norm(output)
        output = self.output_projection(output)

        return output

    def initialize_carry(self, key: jax.Array, input_shape: Tuple[int, ...]) -> Carry:
        """Initialize carry with zero S and z."""
        *batch_dims, _ = input_shape
        S = jnp.zeros((*batch_dims, 1, self.num_heads, self.head_dim, self.head_dim))
        z = jnp.zeros((*batch_dims, 1, self.num_heads, self.head_dim))
        return (S, z)
