import flax.linen as nn
from hydra.utils import instantiate


class QNetwork(nn.Module):
    num_actions: int

    @nn.compact
    def __call__(self, x):
        return nn.Dense(features=num_actions)(x)

