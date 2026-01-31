import flax.linen as nn
import jax.numpy as jnp

from memorax.networks.blocks import FFN
from memorax.networks.identity import Identity


class PatchEmbedding(nn.Module):
    """Converts images to patch sequences via Conv2D."""

    patch_size: int = 16
    features: int = 768

    @nn.compact
    def __call__(self, x: jnp.ndarray, **kwargs) -> jnp.ndarray:
        x = nn.Conv(
            self.features,
            kernel_size=(self.patch_size, self.patch_size),
            strides=(self.patch_size, self.patch_size),
        )(x)
        return x.reshape(x.shape[0], -1, self.features)


class ViT(nn.Module):
    """Vision Transformer feature extractor.

    Can operate in two modes:
    - Token mode (default): For pre-tokenized inputs (e.g., (B, T, num_tokens, token_dim))
    - Image mode: Pass patch_embedding=PatchEmbedding(patch_size, features) to convert images to tokens

    Input shape: (B, T, ...) where T is the time/sequence axis.
    Output shape: (B, T, features)
    """

    features: int = 768
    num_layers: int = 12
    num_heads: int = 12
    expansion_factor: int = 4
    patch_embedding: nn.Module = Identity()

    @nn.compact
    def __call__(self, x: jnp.ndarray, **kwargs) -> jnp.ndarray:
        batch_size, sequence_length, *_ = x.shape
        x = x.reshape(batch_size * sequence_length, *x.shape[2:])

        x = self.patch_embedding(x)
        x = nn.Dense(self.features)(x)

        positional_embedding = self.param(
            "positional_embedding",
            nn.initializers.normal(0.02),
            (1, x.shape[1], self.features),
        )
        x = x + positional_embedding

        for _ in range(self.num_layers):
            skip = x
            x = nn.LayerNorm()(x)
            x = nn.MultiHeadDotProductAttention(num_heads=self.num_heads)(x, x)
            x = skip + x

            skip = x
            x = nn.LayerNorm()(x)
            _, x = FFN(
                features=self.features, expansion_factor=int(self.expansion_factor)
            )(x)
            x = skip + x

        x = nn.LayerNorm()(x)
        x = x.mean(axis=1)

        x = x.reshape(batch_size, sequence_length, -1)
        return x
