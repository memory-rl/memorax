import flax.linen as nn


class Identity(nn.Module):
    @nn.compact
    def __call__(self, x, *args, **kwargs):
        return x
