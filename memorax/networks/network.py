import flax.linen as nn
import jax

from memorax.utils.typing import Array


class Network(nn.Module):
    feature_extractor: nn.Module
    torso: nn.Module
    head: nn.Module

    @nn.compact
    def __call__(
        self,
        observation: Array,
        mask: Array,
        **kwargs,
    ):
        print("Kwargs: ", kwargs)
        x = self.feature_extractor(observation, **kwargs)

        carry, x = self.torso(x, mask=mask, **kwargs)
        return carry, self.head(x, **kwargs)

    @nn.nowrap
    def initialize_carry(self, input_shape):
        key = jax.random.key(0)
        return self.torso.initialize_carry(key, input_shape)
