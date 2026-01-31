from functools import partial
from typing import Callable, Optional

import jax
import flax.linen as nn
import jax.numpy as jnp

from memorax.networks.blocks import FFN
from memorax.networks.identity import Identity
from memorax.networks.sequence_models.utils import get_attention_implementation


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


class Block(nn.Module):
    """Single transformer block with attention and FFN."""

    features: int
    num_heads: int
    expansion_factor: int

    @nn.compact
    def __call__(self, carry, _):
        x = carry
        head_dim = self.features // self.num_heads

        projection = partial(nn.DenseGeneral, features=(self.num_heads, head_dim))

        skip = x
        x = nn.LayerNorm()(x)

        query = projection(name="query")(x)
        key = projection(name="key")(x)
        value = projection(name="value")(x)

        implementation, attention_dtype = get_attention_implementation()
        x = jax.nn.dot_product_attention(
            query.astype(attention_dtype),
            key.astype(attention_dtype),
            value.astype(attention_dtype),
            implementation=implementation,
        ).astype(query.dtype)

        x = nn.DenseGeneral(features=self.features, axis=(-2, -1), name="out")(x)
        x = skip + x

        skip = x
        x = nn.LayerNorm()(x)
        _, x = FFN(features=self.features, expansion_factor=self.expansion_factor)(x)
        x = skip + x

        return x, None


class ViT(nn.Module):
    """Vision Transformer feature extractor."""

    features: int = 768
    num_layers: int = 4
    num_heads: int = 4
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

        x, _ = nn.scan(
            Block,
            variable_axes={"params": 0},
            split_rngs={"params": True},
            length=self.num_layers,
        )(
            features=self.features,
            num_heads=self.num_heads,
            expansion_factor=self.expansion_factor,
        )(x, None)

        x = nn.LayerNorm()(x)
        x = x.mean(axis=1)

        x = x.reshape(batch_size, sequence_length, -1)
        return x
