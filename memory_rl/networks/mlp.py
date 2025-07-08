import flax.linen as nn
import jax.numpy as jnp
from hydra.utils import instantiate, call


class MLP(nn.Module):
    layers: list

    @nn.compact
    def __call__(self, x):
        for layer in self.layers:
            x = instantiate(layer.module, features=layer.module.features, use_bias=layer.module.use_bias, dtype=jnp.float32)(x)
            x = call(layer.activation)(x)
        return x
