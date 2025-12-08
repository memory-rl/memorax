from typing import Callable, Sequence, Union, Optional

import flax.linen as nn
import jax.numpy as jnp


class CNN(nn.Module):

    features: Sequence[int]
    kernel_sizes: Sequence[tuple[int, int]]
    strides: Sequence[tuple[int, int]]
    padding: str = "VALID"
    activation: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    normalizer: Optional[Union[nn.LayerNorm, nn.BatchNorm]] = None
    poolings: Optional[Sequence[Optional[Callable[[jnp.ndarray], jnp.ndarray]]]] = None
    kernel_init: nn.initializers.Initializer = nn.initializers.lecun_normal()
    bias_init: nn.initializers.Initializer = nn.initializers.zeros_init()

    @nn.compact
    def __call__(self, x: jnp.ndarray, **kwargs) -> jnp.ndarray:
        if self.poolings is None:
            self.poolings = [None] * len(self.features)

        for feature, kernel_size, stride, pooling in zip(
            self.features, self.kernel_sizes, self.strides, self.poolings
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
            if pooling is not None:
                x = pooling(x)
        x = x.reshape((x.shape[0], -1))
        return x
