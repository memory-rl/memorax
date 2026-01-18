from functools import partial
from typing import Any, Optional

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax import struct
from flax.linen.module import nowrap
from flax.linen.recurrent import Carry
from flax.typing import Dtype

from memorax.networks.sequence_models.utils import (
    get_attention_implementation, get_input_shape)
from memorax.utils.typing import Array

from .sequence_model import SequenceModel


def _get_positions(mask, start):
    """Generates positions based on episode boundaries."""

    def step(carry, mask):
        next_carry = jnp.where(mask, 0, carry + 1)
        return next_carry, next_carry

    start = start.squeeze(-1)
    next_position, positions = jax.lax.scan(step, start, mask.T)
    return positions.T, next_position[:, None]


@struct.dataclass
class KVCache:
    position: Array
    mask: Array
    key: Array
    value: Array


class MultiHeadAttentionBlock(nn.Module):
    features: int
    num_heads: int
    context_length: int
    dtype: Any
    param_dtype: Any
    kernel_init: Any
    bias_init: Any
    dropout: float = 0.0

    @nn.compact
    def __call__(self, x, mask, kv_cache):
        head_dim = self.features // self.num_heads

        B, T, *_ = x.shape

        assert T <= self.context_length, (
            f"T must be less than or equal to context_length, but was T: {T}, context_length: {self.context_length}"
        )

        projection = partial(
            nn.DenseGeneral,
            features=(self.num_heads, head_dim),
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
        )

        query = projection(name="query")(x)

        key = projection(name="key")(x)
        key = jnp.concatenate([kv_cache.key, key], axis=1)
        key = key[:, -self.context_length :]

        value = projection(name="value")(x)
        value = jnp.concatenate([kv_cache.value, value], axis=1)
        value = value[:, -self.context_length :]

        query_input = (
            jnp.cumsum(mask.astype(jnp.int32), axis=1)
            + jnp.max(jnp.cumsum(kv_cache.mask, axis=1), axis=1)[..., None]
        )

        key_mask = jnp.concatenate([kv_cache.mask, mask], axis=1, dtype=jnp.int32)
        key_input = jnp.cumsum(key_mask, axis=1)
        key_input = key_input[:, -self.context_length :]

        attention_mask = nn.make_attention_mask(
            query_input, key_input, pairwise_fn=jnp.equal
        )

        query_input = jnp.arange(T) + self.context_length
        query_input = jnp.broadcast_to(query_input, (B, T))
        key_input = jnp.arange(self.context_length + T)
        key_input = jnp.broadcast_to(key_input, (B, self.context_length + T))
        key_input = key_input[:, -self.context_length :]
        causal_mask = nn.make_attention_mask(
            query_input, key_input, pairwise_fn=jnp.greater_equal
        )

        B, _, T, S = attention_mask.shape
        attention_mask = jnp.broadcast_to(attention_mask, (B, self.num_heads, T, S))

        B, _, T, S = causal_mask.shape
        causal_mask = jnp.broadcast_to(causal_mask, (B, self.num_heads, T, S))

        x = jax.nn.dot_product_attention(
            query.astype(jnp.bfloat16),
            key.astype(jnp.bfloat16),
            value.astype(jnp.bfloat16),
            mask=nn.combine_masks(attention_mask, causal_mask, dtype=jnp.bool),
            # implementation=get_attention_implementation(),
            implementation="xla",
        )

        y = nn.DenseGeneral(
            self.features,
            axis=(-2, -1),
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
            name="out",
        )(x)

        y = nn.Dropout(rate=self.dropout)(y, deterministic=not self.has_rng("dropout"))

        _, next_position = _get_positions(mask, start=kv_cache.position)
        mask = key_mask[:, -self.context_length :]
        kv_cache = kv_cache.replace(
            position=next_position, mask=mask, key=key, value=value
        )

        return y, kv_cache


class MLP(nn.Module):
    features: int
    hidden_dim: int
    dtype: Any
    param_dtype: Any
    kernel_init: Any
    bias_init: Any
    dropout: float = 0.0

    @nn.compact
    def __call__(self, x):
        projection = partial(
            nn.Dense,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
        )
        x = projection(features=self.hidden_dim, name="up_proj")(x)
        x = jax.nn.gelu(x, approximate=True)
        x = projection(features=self.features, name="down_proj")(x)
        x = nn.Dropout(rate=self.dropout)(x, deterministic=not self.has_rng("dropout"))
        return x


class GPT2Block(nn.Module):
    features: int
    num_heads: int
    hidden_dim: int
    context_length: int
    dtype: Any
    param_dtype: Any
    kernel_init: Any
    bias_init: Any

    @nn.compact
    def __call__(self, x, mask, kv_cache):
        skip = x
        x = nn.LayerNorm(
            epsilon=1e-5,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )(x)
        x, kv_cache = MultiHeadAttentionBlock(
            features=self.features,
            num_heads=self.num_heads,
            context_length=self.context_length,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
        )(x, mask, kv_cache)
        x = x + skip
        skip = x
        x = nn.LayerNorm(
            epsilon=1e-5,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )(x)
        x = MLP(
            features=self.features,
            hidden_dim=self.hidden_dim,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
        )(x)
        x = x + skip
        return x, kv_cache


class GPT2(SequenceModel):
    features: int
    num_embeddings: int
    num_layers: int = 12
    num_heads: int = 12
    hidden_dim: Optional[int] = None
    context_length: int = 1024
    dropout: float = 0.0
    dtype: Dtype | None = None
    param_dtype: Dtype = jnp.float32
    kernel_init: Any = nn.initializers.normal(stddev=0.02)
    bias_init: Any = nn.initializers.zeros_init()

    @property
    def num_feature_axes(self) -> int:
        return 1

    @nowrap
    def initialize_carry(self, key, input_shape):
        batch_size, *_ = input_shape
        head_dim = self.features // self.num_heads
        position = jnp.full((batch_size, 1), -1, dtype=jnp.int32)
        mask = jnp.ones((batch_size, self.context_length), dtype=jnp.int32)
        key = jnp.zeros(
            (batch_size,) + (self.context_length, self.num_heads, head_dim),
            dtype=self.dtype,
        )
        value = jnp.zeros(
            (batch_size,) + (self.context_length, self.num_heads, head_dim),
            dtype=self.dtype,
        )
        return tuple(
            KVCache(position, mask, key, value) for _ in range(self.num_layers)
        )

    def _add_positional_embedding(self, inputs, mask, carry: Carry):
        wpe = nn.Embed(
            num_embeddings=self.num_embeddings,
            features=self.features,
            embedding_init=self.kernel_init,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="wpe",
        )
        kv_cache, *_ = carry

        position, _ = _get_positions(mask, start=kv_cache.position)

        inputs = inputs + wpe(position)
        inputs = nn.Dropout(rate=self.dropout)(
            inputs, deterministic=not self.has_rng("dropout")
        )
        return inputs

    @nn.compact
    def __call__(
        self,
        inputs: jax.Array,
        mask: jax.Array,
        initial_carry: Optional[Carry] = None,
        **kwargs,
    ):
        if initial_carry is None:
            input_shape = get_input_shape(inputs)
            initial_carry = self.initialize_carry(jax.random.key(0), input_shape)

        carry: Carry = initial_carry

        x = self._add_positional_embedding(inputs, mask, carry)

        new_carry = []
        for layer_idx, kv_cache in enumerate(carry):
            x, kv_cache = GPT2Block(
                features=self.features,
                num_heads=self.num_heads,
                hidden_dim=self.hidden_dim or 4 * self.features,
                context_length=self.context_length,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                kernel_init=self.kernel_init,
                bias_init=self.bias_init,
                name=f"layer_{layer_idx}",
            )(
                x,
                mask=mask,
                kv_cache=kv_cache,
            )
            new_carry.append(kv_cache)
        new_carry = tuple(new_carry)

        x = nn.LayerNorm(
            epsilon=1e-5,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )(x)
        return new_carry, x
