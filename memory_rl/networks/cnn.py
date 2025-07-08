import chex
import flax.linen as nn
from hydra.utils import instantiate, call

class CNN(nn.Module):
    layers: list

    @nn.compact
    def __call__(self, x):
        for layer in self.layers:
            x = instantiate(layer.module, features=layer.module.features, kernel_size=layer.module.kernel_size, stride=layer.module.stride, padding=layer.module.padding, use_bias=layer.module.use_bias)(x)
            x = call(layer.activation, x=x)

        return x
