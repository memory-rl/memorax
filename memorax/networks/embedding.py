from typing import Callable, Sequence, Union, Optional

import flax.linen as nn
import jax.numpy as jnp

default_embed_init = nn.initializers.variance_scaling(
  1.0, 'fan_in', 'normal', out_axis=0
)

class Embedding(nn.Module):

    features: int
    num_embeddings: int
    embedding_init: nn.initializers.Initializer = default_embed_init

    @nn.compact
    def __call__(self, x: jnp.ndarray, **kwargs) -> jnp.ndarray:
        x = nn.Embed(self.num_embeddings, self.features, embedding_init=self.embedding_init)(x)
        return x
