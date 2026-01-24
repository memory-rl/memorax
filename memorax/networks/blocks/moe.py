from typing import Optional, Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp

from memorax.networks.sequence_models.utils import get_input_shape
from memorax.utils.typing import Array, Carry

from .base import Block
from .router import TopKRouter


class MoE(nn.Module, Block):
    """Mixture of Experts block for horizontal scaling."""

    experts: Sequence[nn.Module]
    router: TopKRouter

    @nn.compact
    def __call__(
        self,
        inputs: Array,
        mask: Optional[Array] = None,
        initial_carry: Optional[Carry] = None,
        **kwargs,
    ) -> tuple[Carry, Array]:
        if initial_carry is None:
            initial_carry = self.initialize_carry(
                jax.random.key(0), get_input_shape(inputs)
            )

        weights, indices = self.router(inputs)
        batch_size, seq_len, _ = inputs.shape

        outputs, carry = [], []
        for expert, carry_i in zip(self.experts, initial_carry):
            carry_i, x = expert(inputs, mask=mask, initial_carry=carry_i, **kwargs)
            outputs.append(x)
            carry.append(carry_i)

        stacked = jnp.stack(outputs, axis=0)
        batch_indices = jnp.broadcast_to(
            jnp.arange(batch_size)[:, None, None], indices.shape
        )
        sequence_indices = jnp.broadcast_to(
            jnp.arange(seq_len)[None, :, None], indices.shape
        )
        selected = stacked[indices, batch_indices, sequence_indices]
        output = jnp.einsum("bskf,bsk->bsf", selected, weights)

        return tuple(carry), output

    @nn.nowrap
    def initialize_carry(self, key, input_shape) -> Carry:
        return tuple(
            expert.initialize_carry(key, input_shape) for expert in self.experts
        )
