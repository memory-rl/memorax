from typing import Callable, Optional, Sequence

import flax.linen as nn
import jax.numpy as jnp

from memorax.networks.identity import Identity


class MLP(nn.Module):
    features: int | Sequence[int]
    activation: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    normalizer: Callable = Identity
    kernel_init: nn.initializers.Initializer = nn.initializers.lecun_normal()
    bias_init: nn.initializers.Initializer = nn.initializers.zeros_init()

    @nn.compact
    def __call__(self, x: jnp.ndarray, **kwargs) -> jnp.ndarray:
        if isinstance(self.features, int):
            features = [self.features]
        else:
            features = self.features

        for feature in features:
            x = nn.Dense(
                feature, kernel_init=self.kernel_init, bias_init=self.bias_init
            )(x)
            if self.normalizer is not None:
                x = self.normalizer()(x)
            x = self.activation(x)
        return x
