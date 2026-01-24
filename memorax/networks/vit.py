import flax.linen as nn
import jax.numpy as jnp

from memorax.networks.blocks import FFN


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
    """Vision Transformer feature extractor."""

    patch_size: int = 16
    features: int = 768
    num_layers: int = 12
    num_heads: int = 12
    expansion_factor: int = 4

    @nn.compact
    def __call__(self, x: jnp.ndarray, **kwargs) -> jnp.ndarray:
        x = PatchEmbedding(self.patch_size, self.features)(x)

        positional_embeddin = self.param(
            nn.initializers.normal(0.02), (1, x.shape[1], self.features)
        )
        x = x + positional_embeddin

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

        return x
