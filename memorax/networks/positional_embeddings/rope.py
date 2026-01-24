from typing import Any

import jax.numpy as jnp
from flax import struct

from memorax.utils.typing import Array

from .base import RelativePositionalEmbedding


@struct.dataclass
class RoPE(RelativePositionalEmbedding):
    base: float = 10000.0

    def compute_coefficients(self, dim: int, max_seq_len: int) -> Array:
        t = jnp.arange(max_seq_len, dtype=jnp.float32)
        frequencies = jnp.outer(
            t,
            1.0 / (self.base ** (jnp.arange(0, dim, 2, dtype=jnp.float32) / dim)),
        )
        return frequencies

    def rotate(self, x: Array) -> Array:
        x1, x2 = jnp.split(x, 2, axis=-1)
        return jnp.concatenate([-x2, x1], axis=-1)

    def apply(self, x: Array, positions: Array) -> Array:
        head_dim = x.shape[-1]
        t = positions.astype(jnp.float32)
        freqs = 1.0 / (self.base ** (jnp.arange(0, head_dim, 2, dtype=jnp.float32) / head_dim))
        frequencies = t[..., None] * freqs
        frequencies = frequencies[:, :, None, :]

        cos = jnp.cos(frequencies)
        sin = jnp.sin(frequencies)

        cos = jnp.concatenate([cos, cos], axis=-1)
        sin = jnp.concatenate([sin, sin], axis=-1)

        return x * cos + self.rotate(x) * sin

    def __call__(
        self, query: Array, key: Array, query_pos: Array, key_pos: Array
    ) -> tuple[Array, Array, Any]:
        query = self.apply(query, query_pos)
        key = self.apply(key, key_pos)
        return query, key, None
