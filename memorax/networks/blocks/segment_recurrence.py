from typing import Any, Optional

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax import struct

from memorax.networks.blocks import Block
from memorax.networks.sequence_models.sequence_model import SequenceModel
from memorax.networks.sequence_models.utils import get_input_shape
from memorax.utils.typing import Array, Carry


@struct.dataclass
class Memory:
    state: Array
    mask: Array


class SegmentRecurrence(nn.Module, Block):
    """Wraps a sequence model with segment-level recurrence memory.

    This block maintains a fixed-length memory of past outputs that can be
    used by the wrapped sequence model for cross-segment attention.

    Args:
        sequence_model: The underlying sequence model to wrap.
        memory_length: Maximum number of past timesteps to retain.
        features: Feature dimension of the memory.
        dtype: Data type for memory storage.
    """

    sequence_model: SequenceModel
    memory_length: int
    features: int
    dtype: Any = None

    @nn.nowrap
    def initialize_carry(self, key, input_shape) -> Carry:
        batch_size, *_ = input_shape
        state = jnp.zeros((batch_size, 0, self.features), dtype=self.dtype)
        mask = jnp.zeros((batch_size, 0), dtype=jnp.int32)
        memory = Memory(state=state, mask=mask)

        carry = self.sequence_model.initialize_carry(key, input_shape)
        return (memory, carry)

    @nn.compact
    def __call__(
        self,
        inputs: Array,
        mask: Optional[Array] = None,
        initial_carry: Optional[Carry] = None,
        **kwargs,
    ) -> tuple[Carry, Array]:
        if initial_carry is None:
            input_shape = get_input_shape(inputs)
            initial_carry = self.initialize_carry(jax.random.key(0), input_shape)

        if mask is None:
            batch_size, seq_len, *_ = inputs.shape
            mask = jnp.zeros((batch_size, seq_len), dtype=jnp.int32)

        memory, carry = initial_carry

        carry, y = self.sequence_model(
            inputs,
            mask,
            initial_carry=carry,
            memory=memory.state,
            memory_mask=memory.mask,
            **kwargs,
        )

        state = jnp.concatenate([memory.state, jax.lax.stop_gradient(y)], axis=1)
        state = state[:, -self.memory_length :]

        mask = jnp.concatenate([memory.mask, mask], axis=1)
        mask = mask[:, -self.memory_length :]

        memory = Memory(state=state, mask=mask)

        return (memory, carry), y
