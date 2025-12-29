from abc import ABC, abstractmethod

import flax.linen as nn

from memorax.utils.typing import Array, Carry

class SequenceModel(ABC, nn.Module):

    @abstractmethod
    def __call__(
        self,
        inputs: Array,
        mask: Array,
        initial_carry: Carry,
        **kwargs,
    ):
        pass

    @abstractmethod
    def initialize_carry(self, key, input_shape):
        pass
