from typing import Any

import jax.numpy as jnp
from flax import struct

from memorax.utils.typing import Array

from .base import PositionalEmbedding


@struct.dataclass
class ALiBi(PositionalEmbedding):
    num_heads: int

    def compute_coefficients(self) -> Array:
        assert (
            self.num_heads & (self.num_heads - 1) == 0
        ), "num_heads must be a power of 2"
        ratio = 2 ** (-8 / self.num_heads)
        return ratio ** jnp.arange(1, self.num_heads + 1)

    def apply(self, query_pos: Array, key_pos: Array) -> Array:
        slopes = self.compute_coefficients()
        relative_pos = query_pos[:, :, None] - key_pos[:, None, :]
        return slopes[None, :, None, None] * relative_pos[:, None, :, :]

    def __call__(
        self, query: Array, key: Array, query_pos: Array, key_pos: Array
    ) -> tuple[Array, Array, Any]:
        bias = self.apply(query_pos, key_pos)
        return query, key, bias
