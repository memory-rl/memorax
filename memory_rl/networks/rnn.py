import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.linen.recurrent import RNNCellBase

from memory_rl.networks.recurrent import MaskedRNN


class RNN(nn.Module):
    cell: RNNCellBase

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        mask: jnp.ndarray,
        initial_carry: jnp.ndarray | None = None,
        return_carry_history: bool = False,
        **kwargs,
    ):
        hidden_state, x = MaskedRNN(self.cell, return_carry=True)(
            x,
            mask,
            initial_carry=initial_carry,
            return_carry_history=return_carry_history,
        )
        return hidden_state, x

    def initialize_carry(self, input_shape):
        return self.cell.initialize_carry(jax.random.key(0), input_shape)
