from typing import Optional, Sequence

import flax.linen as nn

from memorax.utils.typing import Array, Carry

from .base import Block


class Stack(nn.Module, Block):
    """Vertically stacks multiple heterogeneous blocks.

    Each block's output becomes the next block's input. Carry states are
    maintained per-block as a tuple, allowing different block types with
    different carry structures to be composed.

    Args:
        blocks: Sequence of blocks to stack. Each must implement the Block protocol.

    Example:
        stack = Stack(blocks=[
            Residual(module=PreNorm(module=SelfAttention(...))),
            Residual(module=PreNorm(module=FFN(...))),
            Residual(module=PreNorm(module=SelfAttention(...))),
            Residual(module=PreNorm(module=FFN(...))),
        ])
        carry, output = stack(inputs, mask, initial_carry)
    """

    blocks: Sequence[nn.Module]

    @nn.compact
    def __call__(
        self,
        inputs: Array,
        mask: Optional[Array] = None,
        initial_carry: Optional[tuple[Carry, ...]] = None,
        **kwargs,
    ) -> tuple[tuple[Carry, ...], Array]:
        if initial_carry is None:
            initial_carry = tuple(None for _ in self.blocks)

        x = inputs
        carries = []

        for i, block in enumerate(self.blocks):
            carry, x = block(x, mask=mask, initial_carry=initial_carry[i], **kwargs)
            carries.append(carry)

        return tuple(carries), x

    @nn.nowrap
    def initialize_carry(self, key, input_shape):
        carries = []
        for block in self.blocks:
            if hasattr(block, "initialize_carry"):
                carries.append(block.initialize_carry(key, input_shape))
            else:
                carries.append(None)
        return tuple(carries)
