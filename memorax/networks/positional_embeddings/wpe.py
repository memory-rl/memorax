from typing import Optional

import jax
import jax.numpy as jnp
from flax import linen as nn

from memorax.utils.typing import Array, Carry

from .base import AbsolutePositionalEmbedding


class LearnablePositionalEmbedding(AbsolutePositionalEmbedding, nn.Module):
    num_embeddings: int
    features: int

    def initialize_carry(self, batch_size: int) -> Carry:
        return jnp.zeros((batch_size,), dtype=jnp.int32)

    @nn.compact
    def __call__(
        self,
        inputs: Array,
        mask: Array,
        initial_carry: Optional[Carry] = None,
        **kwargs,
    ) -> tuple[Array, Carry]:
        batch_size = inputs.shape[0]

        if initial_carry is None:
            initial_carry = self.initialize_carry(batch_size)

        def step(position: Array, mask: Array) -> tuple[Array, Array]:
            next_position = jnp.where(mask, 0, position + 1)
            return next_position, position

        def compute_positions(mask: Array, offset: Array) -> tuple[Array, Array]:
            carry, positions = jax.lax.scan(step, offset, mask)
            return positions, carry

        positions, carry = jax.vmap(compute_positions)(mask, initial_carry)

        position_embeddings = nn.Embed(
            num_embeddings=self.num_embeddings, features=self.features
        )(positions)

        return inputs + position_embeddings, carry
