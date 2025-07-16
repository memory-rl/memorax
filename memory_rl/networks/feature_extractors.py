from typing import Callable, Sequence

import flax.linen as nn
import jax.numpy as jnp


class MLP(nn.Module):

    features: Sequence[int]
    activation: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        for feature in self.features:
            x = nn.Dense(feature)(x)
            x = self.activation(x)
        return x


class CNN(nn.Module):

    features: Sequence[int]
    kernel_sizes: Sequence[tuple[int, int]]
    strides: Sequence[tuple[int, int]]
    activation: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        for feature, kernel_size, stride in zip(
            self.features, self.kernel_sizes, self.strides
        ):
            x = nn.Conv(feature, kernel_size=kernel_size, strides=stride)(x)
            x = self.activation(x)
        return x
