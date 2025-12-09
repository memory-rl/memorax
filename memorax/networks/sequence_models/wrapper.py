import jax.numpy as jnp
import flax.linen as nn

class RecurrentWrapper(nn.Module):
    network: nn.Module

    def __call__(self, inputs, mask, initial_carry=None, **kwargs):
        carry = initial_carry
        return carry, self.network(inputs)

    def initialize_carry(self, rng, input_shape):
        return jnp.zeros(input_shape)

