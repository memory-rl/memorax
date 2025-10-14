from typing import Optional

import flax.linen as nn
import jax
import jax.numpy as jnp


class Network(nn.Module):
    feature_extractor: nn.Module
    torso: nn.Module
    head: nn.Module

    @nn.compact
    def __call__(
        self,
        observation: jnp.ndarray,
        action: Optional[jnp.ndarray] = None,
        reward: Optional[jnp.ndarray] = None,
        done: Optional[jnp.ndarray] = None,
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
        observation: jnp.ndarray,
        mask: jnp.ndarray,
        action: Optional[jnp.ndarray] = None,
        reward: Optional[jnp.ndarray] = None,
        done: Optional[jnp.ndarray] = None,
        **kwargs,
    ):
        x = self.feature_extractor(
            observation, action=action, reward=reward, done=done, **kwargs
        )

        carry, x = self.torso(x, mask=mask, **kwargs)
        # hidden_state, x = MaskedRNN(
        #     self.torso,
        #     time_major=False,
        #     unroll=16,
        #     return_carry=True,
        #     split_rngs={"params": False, "memory": True, "dropout": True},
        #     variable_broadcast={"params", "constants"},
        # )(
        #     x,
        #     mask,
        #     **kwargs,
        # )
        return carry, self.head(x, **kwargs)

    def initialize_carry(self, input_shape):
        key = jax.random.key(0)
        return self.torso.initialize_carry(key, input_shape)
