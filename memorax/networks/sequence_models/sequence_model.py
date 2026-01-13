from abc import ABC, abstractmethod
from typing import Optional

from memorax.utils.typing import Array, Carry


class SequenceModel(ABC):
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
