from typing import Optional

import jax
import jax.numpy as jnp
from flax import linen as nn


def _check_positions(positions, num_embeddings):
    if not jnp.all(positions < num_embeddings):
        raise ValueError(
            f"Position indices exceed num_embeddings ({num_embeddings}). "
            f"Max position: {jnp.max(positions)}. "
            "Ensure num_embeddings >= context_length or that episode resets occur frequently enough."
        )

from memorax.utils.typing import Array, Carry

from .base import AbsolutePositionalEmbedding


class LearnablePositionalEmbedding(AbsolutePositionalEmbedding, nn.Module):
    num_embeddings: int
    features: int

    @nn.nowrap
    def initialize_carry(self, key, input_shape) -> Carry:
        *batch_dims, _ = input_shape
        return jnp.zeros(batch_dims, dtype=jnp.int32)

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
            initial_carry = self.initialize_carry(None, (batch_size, inputs.shape[-1]))

        def step(position: Array, mask: Array) -> tuple[Array, Array]:
            next_position = jnp.where(mask, 0, position + 1)
            return next_position, position

        def compute_positions(mask: Array, offset: Array) -> tuple[Array, Array]:
            carry, positions = jax.lax.scan(step, offset, mask)
            return positions, carry

        positions, carry = jax.vmap(compute_positions)(mask, initial_carry)

        jax.debug.callback(_check_positions, positions, self.num_embeddings)

        position_embeddings = nn.Embed(
            num_embeddings=self.num_embeddings, features=self.features
        )(positions)

        return carry, inputs + position_embeddings
