from abc import ABC, abstractmethod
from typing import Optional, Tuple

import jax
from flax import linen as nn

from memorax.utils.typing import Array, Carry


class SequenceModel(ABC, nn.Module):
    @abstractmethod
    def __call__(
        self,
        inputs: Array,
        mask: Array,
        initial_carry: Optional[Carry] = None,
        **kwargs,
    ) -> tuple: ...

    @abstractmethod
    def initialize_carry(self, key, input_shape) -> Carry: ...

    def local_jacobian(
        self,
        inputs: Array,
        mask: Array,
        carry: Carry,
        sensitivity: Optional[dict] = None,
        **kwargs,
    ) -> Tuple[Carry, Array, Optional[dict]]:
        """Forward pass with optional RTRL sensitivity propagation.

        Returns (next_carry, y, next_sensitivity).
        By default, calls __call__ and returns None sensitivity.
        Override to provide RTRL support.
        """
        next_carry, y = self(inputs, mask, carry, **kwargs)
        return next_carry, y, None

    def initialize_sensitivity(
        self, key: jax.Array, input_shape: Tuple[int, ...]
    ) -> Optional[dict]:
        """Initialize RTRL sensitivity tensors. Returns None if unsupported."""
        return None
