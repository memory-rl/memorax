from typing import Optional, Tuple

import jax
from flax import linen as nn

from memorax.utils.typing import Array, Carry

from .sequence_model import SequenceModel
from .utils import get_input_shape


class RTRL(SequenceModel):
    sequence_model: SequenceModel

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
            initial_carry = self.initialize_carry(jax.random.key(0), input_shape)

        carry, sensitivity = initial_carry
        assert sensitivity is not None, (
            f"{type(self.sequence_model).__name__} does not support RTRL. "
            "Ensure the inner model implements local_jacobian and initialize_sensitivity."
        )

        next_carry, y, next_sensitivity = self.sequence_model.local_jacobian(
            inputs, mask, carry, sensitivity=sensitivity, **kwargs
        )

        return (next_carry, next_sensitivity), y

    def initialize_carry(self, key: jax.Array, input_shape: Tuple[int, ...]) -> Carry:
        carry = self.sequence_model.initialize_carry(key, input_shape)
        sensitivity = self.sequence_model.initialize_sensitivity(key, input_shape)
        return (carry, sensitivity)
