from typing import Callable, Sequence, Union, Optional

import flax.linen as nn
import jax.numpy as jnp

import jax


class MLP(nn.Module):

    features: Sequence[int]
    activation: Callable[[jnp.ndarray], jnp.ndarray] = nn.tanh
    normalizer: Optional[Union[nn.LayerNorm, nn.BatchNorm]] = None
    kernel_init: nn.initializers.Initializer = nn.initializers.lecun_normal()
    bias_init: nn.initializers.Initializer = nn.initializers.zeros_init()

    @nn.compact
    def __call__(self, x: jnp.ndarray, **kwargs) -> jnp.ndarray:
        for feature in self.features:
            x = nn.Dense(
                feature, kernel_init=self.kernel_init, bias_init=self.bias_init
            )(x)
            if self.normalizer is not None:
                x = self.normalizer(x)
            x = self.activation(x)
        return x
