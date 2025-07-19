from typing import Callable, Optional, Sequence
import jax

import flax.linen as nn
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
        temperature: float = 1.0,
    ):
        x = (
            jnp.concatenate([observation, action], axis=-1)
            if action is not None
            else observation
        )
        x = self.feature_extractor(x)
        x = self.torso(x)
        return self.head(x, temperature=temperature)


class RecurrentNetwork(nn.Module):
    feature_extractor: nn.Module
    cell: nn.RNNCellBase
    head: nn.Module

    @nn.compact
    def __call__(
        self,
        observation: jnp.ndarray,
        mask: jnp.ndarray,
        action: Optional[jnp.ndarray] = None,
        initial_carry: Optional[jnp.ndarray] = None,
        return_carry_history: bool = False,
        temperature: float = 1.0,
    ):
        x = (
            jnp.concatenate([observation, action], axis=-1)
            if action is not None
            else observation
        )
        x = self.feature_extractor(x)
        hidden_state, x = MaskedRNN(self.cell, return_carry=True)(
            x,
            mask,
            initial_carry=initial_carry,
            return_carry_history=return_carry_history,
        )
        return hidden_state, self.head(x, temperature=temperature)

    def initialize_carry(self, input_shape):
        key = jax.random.key(0)
        return self.cell.initialize_carry(key, input_shape)
