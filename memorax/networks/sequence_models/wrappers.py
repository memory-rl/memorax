import flax.linen as nn
import jax
import jax.numpy as jnp
from flax import struct

from .sequence_model import SequenceModel


class SequenceModelWrapper(SequenceModel, nn.Module):
    network: nn.Module

    def __call__(self, inputs, mask, initial_carry=None, **kwargs):
        carry = initial_carry
        return carry, self.network(inputs, **kwargs)

    def initialize_carry(self, key, input_shape):
        batch_size, _ = input_shape
        return jnp.zeros((batch_size, 1))


@struct.dataclass
class MetaMaskState:
    carry: jnp.ndarray
    step: jnp.ndarray


class MetaMaskWrapper(SequenceModel, nn.Module):
    sequence_model: nn.Module
    steps_per_trial: int

    def __call__(self, inputs, mask, initial_carry=None, **kwargs):
        _, sequence_length, *_ = inputs.shape

        if initial_carry is None:
            initial_carry = self.initialize_carry(jax.random.key(0), inputs.shape)

        time_indices = jnp.arange(sequence_length)

        steps = initial_carry.step[:, None] + time_indices[None, :]

        mask = steps % self.steps_per_trial != 0

        carry, outputs = self.sequence_model(inputs, mask, initial_carry.carry)
        carry = MetaMaskState(carry=carry, step=initial_carry.step + sequence_length)
        return carry, outputs

    def initialize_carry(self, key, input_shape):
        batch_size, *_ = input_shape
        return MetaMaskState(
            carry=self.sequence_model.initialize_carry(key, input_shape),
            step=jnp.zeros((batch_size,), dtype=jnp.int32),
        )
