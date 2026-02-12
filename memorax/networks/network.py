from typing import Optional

import flax.linen as nn
import jax

from memorax.networks import Identity
from memorax.utils.typing import Array


class Network(nn.Module):
    feature_extractor: nn.Module = Identity()
    torso: nn.Module = Identity()
    head: nn.Module = Identity()
    auxiliary_losses: dict[str, nn.Module] | None = None

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

        output, aux = self.head(x, action=action, reward=reward, done=done, observation=observation, **kwargs)
        auxiliary_losses = {name: h(x, action=action, reward=reward, done=done, observation=observation, **kwargs) for name, h in (self.auxiliary_losses or {}).items()}
        return carry, (output, {**aux, "auxiliary_losses": auxiliary_losses})

    @nn.nowrap
    def auxiliary_loss(self, aux, transitions):
        return sum(
            (self.auxiliary_losses[name].loss(out, head_aux, transitions)
             for name, (out, head_aux) in aux["auxiliary_losses"].items()),
            0.0,
        )

    @nn.nowrap
    def initialize_carry(self, input_shape):
        key = jax.random.key(0)
        return getattr(self.torso, "initialize_carry", lambda k, s: None)(key, input_shape)
