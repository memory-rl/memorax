from typing import Callable, Sequence

import flax.linen as nn
import jax.numpy as jnp


class CNN(nn.Module):

    features: Sequence[int]
    kernel_sizes: Sequence[tuple[int, int]]
    strides: Sequence[tuple[int, int]]
    activation: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    kernel_init: nn.initializers.Initializer = nn.initializers.lecun_normal()
    bias_init: nn.initializers.Initializer = nn.initializers.zeros_init()

    @nn.compact
    def __call__(self, x: jnp.ndarray, **kwargs) -> jnp.ndarray:
        for feature, kernel_size, stride in zip(
            self.features, self.kernel_sizes, self.strides
        ):
            x = nn.Conv(
                feature,
                kernel_size=kernel_size,
                strides=stride,
                kernel_init=self.kernel_init,
                bias_init=self.bias_init,
            )(x)
            x = self.activation(x)
        return x
