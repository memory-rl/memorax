from typing import Optional

import flax.linen as nn
from flax.linen.recurrent import Carry
import jax

from memory_rl.utils.typing import Array


class Network(nn.Module):
    feature_extractor: nn.Module
    torso: nn.Module
    head: nn.Module

    @nn.compact
    def __call__(
        self,
        observation: Array,
        action: Optional[Array] = None,
        reward: Optional[Array] = None,
        done: Optional[Array] = None,
        **kwargs,
    ):
        x = self.feature_extractor(
            observation, action=action, reward=reward, done=done, **kwargs
        )
        x = self.torso(x, **kwargs)
        return self.head(x, **kwargs)


class SequenceNetwork(nn.Module):
    feature_extractor: nn.Module
    torso: nn.Module
    head: nn.Module

    @nn.compact
    def __call__(
        self,
        observation: Array,
        mask: Array,
        action: Optional[Array] = None,
        reward: Optional[Array] = None,
        initial_carry: Carry | None = None,
        **kwargs,
    ):
        x = self.feature_extractor(observation, action=action, reward=reward, **kwargs)

        carry, x = self.torso(x, mask=mask, initial_carry=initial_carry, **kwargs)
        return carry, self.head(x, **kwargs)

    @nn.nowrap
    def initialize_carry(self, input_shape):
        key = jax.random.key(0)
        return self.torso.initialize_carry(key, input_shape)
