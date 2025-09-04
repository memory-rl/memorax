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
        temperature: float = 1.0,
    ):
        x = self.feature_extractor(observation, action=action, reward=reward, done=done)
        x = self.torso(x)
        return self.head(x, temperature=temperature)


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
        initial_carry: Optional[jnp.ndarray] = None,
        return_carry_history: bool = False,
        temperature: float = 1.0,
    ):
        x = self.feature_extractor(observation, action=action, reward=reward, done=done)
        hidden_state, x = MaskedRNN(self.torso, time_major=False, unroll=16, return_carry=True, split_rngs={"params": False, "memory": True})(
            x,
            mask,
            initial_carry=initial_carry,
            return_carry_history=return_carry_history,
        )
        return hidden_state, self.head(x, temperature=temperature)

    def initialize_carry(self, input_shape):
        key = jax.random.key(0)
        return self.torso.initialize_carry(key, input_shape)
