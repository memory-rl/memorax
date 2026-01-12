from typing import Callable, Optional, Sequence

import flax.linen as nn
import jax.numpy as jnp


class MLP(nn.Module):

    features: Sequence[int]
    activation: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    normalizer: Optional[Callable] = None
    kernel_init: nn.initializers.Initializer = nn.initializers.lecun_normal()
    bias_init: nn.initializers.Initializer = nn.initializers.zeros_init()

    @property
    def layers(self) -> Sequence[int]:
        if isinstance(self.features, int):
            return (self.features,)
        return self.features

    @nn.compact
    def __call__(self, x: jnp.ndarray, **kwargs) -> jnp.ndarray:
        for feature in self.layers:
            x = nn.Dense(
                feature, kernel_init=self.kernel_init, bias_init=self.bias_init
            )(x)
            if self.normalizer is not None:
                x = self.normalizer()(x)
            x = self.activation(x)
        return x
