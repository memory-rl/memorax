from typing import Optional

import flax.linen as nn
import jax

from memorax.utils.typing import Array


class Network(nn.Module):
    feature_extractor: nn.Module
    torso: Optional[nn.Module]
    head: nn.Module

    @nn.compact
    def __call__(
        self,
        observation: Array,
        mask: Array,
        **kwargs,
    ):
        x = self.feature_extractor(observation, **kwargs)

        carry = None
        if self.torso is not None:
            carry, x = self.torso(x, mask=mask, **kwargs)

        return carry, self.head(x, **kwargs)

    @nn.nowrap
    def initialize_carry(self, input_shape):
        key = jax.random.key(0)
        carry = None
        if self.torso is not None:
            carry = self.torso.initialize_carry(key, input_shape)
        return carry
