from typing import Callable, Optional, Sequence, Union

import flax.linen as nn
import jax.numpy as jnp


class CNN(nn.Module):

    features: Sequence[int]
    kernel_sizes: Sequence[tuple[int, int]]
    strides: Sequence[int | tuple[int, int]]
    padding: str = "VALID"
    activation: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    normalizer: Optional[Callable] = None
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
                padding=self.padding,
                kernel_init=self.kernel_init,
                bias_init=self.bias_init,
            )(x)
            if self.normalizer is not None:
                x = self.normalizer()(x)
            x = self.activation(x)

        batch_size, sequence_length, *_ = x.shape
        x = x.reshape((batch_size, sequence_length, -1))
        return x
