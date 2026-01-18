from typing import Optional

import flax.linen as nn
import jax

from memorax.networks import Identity
from memorax.networks.sequence_models.wrappers import SequenceModelWrapper
from memorax.utils.typing import Array


class Network(nn.Module):
    feature_extractor: nn.Module = Identity()
    torso: nn.Module = SequenceModelWrapper(Identity())
    head: nn.Module = Identity()

    @nn.compact
    def __call__(
        self,
        observation: Array,
        mask: Array,
        **kwargs,
    ):
        x = observation

        x = self.feature_extractor(observation, **kwargs)

        carry, x = self.torso(x, mask=mask, **kwargs)

        x = self.head(x, **kwargs)
        return carry, x

    @nn.nowrap
    def initialize_carry(self, input_shape):
        key = jax.random.key(0)
        carry = None
        if self.torso is not None:
            carry = self.torso.initialize_carry(key, input_shape)
        return carry
