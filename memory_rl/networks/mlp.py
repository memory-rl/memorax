from typing import Callable, Sequence

import flax.linen as nn
import jax.numpy as jnp


class MLP(nn.Module):

    features: Sequence[int]
    activation: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    kernel_init: nn.initializers.Initializer = nn.initializers.lecun_normal()
    bias_init: nn.initializers.Initializer = nn.initializers.zeros_init()

    @nn.compact
    def __call__(self, x: jnp.ndarray, **kwargs) -> jnp.ndarray:
        for feature in self.features:
            x = nn.Dense(
                feature, kernel_init=self.kernel_init, bias_init=self.bias_init
            )(x)
            x = self.activation(x)
        return x
