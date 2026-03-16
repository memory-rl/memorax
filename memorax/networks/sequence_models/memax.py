from typing import Optional, Tuple

import jax
import jax.numpy as jnp
from flax import linen as nn

from memorax.utils.axes import get_input_shape
from memorax.utils.typing import Array, Carry

from .sequence_model import SequenceModel


class MemaxWrapper(SequenceModel, nn.Module):
    model: nn.Module

    def __call__(
        self,
        inputs: Array,
        done: Array,
        initial_carry: Optional[Carry] = None,
        **kwargs,
    ) -> Tuple[Carry, Array]:
        if initial_carry is None:
            initial_carry = self.initialize_carry(
                jax.random.key(0), get_input_shape(inputs)
            )

        start = jnp.concatenate(
            [jnp.zeros_like(done[:, :1]), done[:, :-1]], axis=1
        ).astype(bool)

        def apply(carry, sequence, start_flags):
            return self.model(carry, (sequence, start_flags))

        all_carries, output = jax.vmap(apply)(initial_carry, inputs, start)

        carry = jax.tree.map(lambda x: x[:, -1], all_carries)

        return carry, output

    def initialize_carry(self, key, input_shape):
        batch_size = input_shape[0]
        single_carry = self.model.initialize_carry(key)
        return jax.tree.map(
            lambda x: jnp.repeat(jnp.expand_dims(x, 0), batch_size, axis=0),
            single_carry,
        )
