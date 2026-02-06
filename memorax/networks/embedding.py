from typing import Callable, Optional, Sequence, Union

import flax.linen as nn
import jax.numpy as jnp

default_embed_init = nn.initializers.variance_scaling(
    1.0, "fan_in", "normal", out_axis=0
)


class Embedding(nn.Module):
    features: int
    num_embeddings: int
    embedding_init: nn.initializers.Initializer = default_embed_init

    @nn.compact
    def __call__(self, x: jnp.ndarray, **kwargs) -> jnp.ndarray:
        x = nn.Embed(
            self.num_embeddings, self.features, embedding_init=self.embedding_init
        )(x)
        return x


class TokenEmbedding(nn.Module):
    """Embeds discrete token observations into a continuous sequence.

    Each token has `num_features` discrete uint8 columns. Each column gets its
    own ``nn.Embed(num_embeddings, features)`` table.  The per-column embeddings
    are concatenated, producing output shape
    ``(batch, num_tokens, num_features * features)``.

    This is intended as the ``patch_embedding`` parameter of :class:`ViT`.
    """

    features: int
    num_features: int
    num_embeddings: int = 256
    embedding_init: nn.initializers.Initializer = default_embed_init

    @nn.compact
    def __call__(self, x: jnp.ndarray, **kwargs) -> jnp.ndarray:
        parts = []
        for i in range(self.num_features):
            col = x[..., i].astype(jnp.int32)
            emb = nn.Embed(
                self.num_embeddings,
                self.features,
                embedding_init=self.embedding_init,
                name=f"embed_{i}",
            )(col)
            parts.append(emb)
        return jnp.concatenate(parts, axis=-1)
