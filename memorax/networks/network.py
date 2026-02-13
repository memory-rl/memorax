from typing import Optional

import flax.linen as nn
import jax

from memorax.networks import Identity
from memorax.utils.typing import Array


class Network(nn.Module):
    feature_extractor: nn.Module = Identity()
    torso: nn.Module = Identity()
    head: nn.Module = Identity()

    @nn.compact
    def __call__(
        self,
        observation: Array,
        mask: Array,
        action: Array,
        reward: Array,
        done: Array,
        initial_carry: Optional[Array] = None,
        **kwargs,
    ):
        x, embeddings = self.feature_extractor(
            observation, action=action, reward=reward, done=done
        )

        match self.torso(
            x,
            mask=mask,
            action=action,
            reward=reward,
            done=done,
            initial_carry=initial_carry,
            **embeddings,
        ):
            case (carry, x):
                pass
            case x:
                carry = None

        x = self.head(x, action=action, reward=reward, done=done, **kwargs)
        return carry, x

    @nn.nowrap
    def initialize_carry(self, input_shape):
        key = jax.random.key(0)
        return getattr(self.torso, "initialize_carry", lambda k, s: None)(key, input_shape)
