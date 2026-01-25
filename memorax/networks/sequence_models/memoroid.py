from abc import abstractmethod
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
from flax import linen as nn

from memorax.utils.typing import Array, Carry

from .sequence_model import SequenceModel
from .utils import broadcast_mask, get_input_shape


class MemoroidCellBase(nn.Module):
    @abstractmethod
    def __call__(self, x: Array, **kwargs) -> Carry: ...

    @abstractmethod
    def binary_operator(self, a: Carry, b: Carry) -> Carry: ...

    @abstractmethod
    def read(self, h: Carry, x: Array, **kwargs) -> Array: ...

    @abstractmethod
    def initialize_carry(
        self, key: jax.Array, input_shape: Tuple[int, ...]
    ) -> Carry: ...


class Memoroid(SequenceModel):
    cell: MemoroidCellBase

    @nn.compact
    def __call__(
        self,
        inputs: Array,
        mask: Array,
        initial_carry: Optional[Carry] = None,
        **kwargs,
    ) -> Tuple[Carry, Array]:
        if initial_carry is None:
            input_shape = get_input_shape(inputs)
            initial_carry = self.cell.initialize_carry(jax.random.key(0), input_shape)

        z = self.cell(inputs, **kwargs)

        z = jax.tree.map(
            lambda c, e: jnp.concatenate([c, e], axis=1),
            initial_carry,
            z,
        )

        reset = jnp.concatenate([jnp.zeros((mask.shape[0], 1)), mask], axis=1)
        reset = reset[..., None]

        @jax.vmap
        def binary_operator(lhs, rhs):
            lhs_i, rhs_i = lhs
            lhs_j, rhs_j = rhs

            combined = self.cell.binary_operator(lhs_i, lhs_j)

            out = jax.tree.map(
                lambda lj, c: jnp.where(broadcast_mask(rhs_j, lj), lj, c),
                lhs_j,
                combined,
            )

            return out, jnp.maximum(rhs_i, rhs_j)

        h, _ = jax.lax.associative_scan(binary_operator, (z, reset), axis=1)

        next_carry = jax.tree.map(lambda s: s[:, -1:], h)
        h = jax.tree.map(lambda s: s[:, 1:], h)

        y = self.cell.read(h, inputs, **kwargs)

        return next_carry, y

    def initialize_carry(self, key: jax.Array, input_shape: Tuple[int, ...]) -> Carry:
        return self.cell.initialize_carry(key, input_shape)
