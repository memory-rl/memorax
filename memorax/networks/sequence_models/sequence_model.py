from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

from flax import struct

from memorax.utils.typing import Array, Carry


class SequenceModel:
    features: int

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
