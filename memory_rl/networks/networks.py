from typing import Optional

import flax.linen as nn
import jax
import jax.numpy as jnp

from memory_rl.networks.recurrent import MaskedRNN


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


class RecurrentNetwork(nn.Module):
    feature_extractor: nn.Module
    torso: nn.RNNCellBase
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

        hidden_state, x = MaskedRNN(
            self.torso,
            time_major=False,
            unroll=16,
            return_carry=True,
            split_rngs={"params": False, "memory": True, "dropout": True},
            variable_broadcast={"params", "constants"},
        )(
            x,
            mask,
            **kwargs,
        )
        return hidden_state, self.head(x, **kwargs)

    def initialize_carry(self, input_shape):
        key = jax.random.key(0)
        return self.torso.initialize_carry(key, input_shape)
