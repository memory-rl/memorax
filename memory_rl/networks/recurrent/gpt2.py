from typing import Any, Optional
from functools import partial
from flax.linen.module import nowrap
from flax.typing import Dtype
import jax.numpy as jnp
import jax
from flax import linen as nn
from flax import struct
from flax.linen.recurrent import Carry

from memory_rl.networks.recurrent.utils import (
    get_attention_implementation,
)
from memory_rl.utils.typing import Array

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
        value = projection(name="value")(x)

        key = jnp.concatenate([kv_cache.key, key], axis=1)
        value = jnp.concatenate([kv_cache.value, value], axis=1)

        query_mask = mask.astype(jnp.int32)
        query_input = jax.lax.cumsum(query_mask, axis=1, reverse=True) 

        key_mask = jnp.concatenate([kv_cache.mask, mask], axis=1, dtype=jnp.int32)
        key_input = jax.lax.cumsum(key_mask, axis=1, reverse=True)

        attention_mask = nn.make_attention_mask(
            query_input, key_input, dtype=jnp.bool_, pairwise_fn=jnp.equal
        )
        B, _, T, S = attention_mask.shape
        attention_mask = jnp.broadcast_to(attention_mask, (B, self.num_heads, T, S))

        x = jax.nn.dot_product_attention(
            query.astype(jnp.bfloat16),
            key.astype(jnp.bfloat16),
            value.astype(jnp.bfloat16),
            is_causal=True,
            mask=attention_mask,
            implementation=get_attention_implementation(),
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

        position = kv_cache.position + jnp.sum(mask, axis=1, dtype=jnp.int32)[:, None] & self.context_length
        mask = key_mask[:, -self.context_length:]
        key = key[:, -self.context_length:, :]
        value = value[:, -self.context_length:, :]
        kv_cache = kv_cache.replace(position=position, mask=mask, key=key, value=value)

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
    def __call__(self, x, mask, kv_cache=None):
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


class GPT2(nn.Module):

    features: int
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
    def initialize_carry(self, rng, input_shape):
        batch_size, *_ = input_shape
        head_dim = self.features // self.num_heads
        position = jnp.zeros((batch_size, 1), dtype=jnp.int32)
        mask = jnp.ones((batch_size, self.context_length), dtype=jnp.int32)
        key = jnp.zeros(
            (batch_size,) + (self.context_length, self.num_heads, head_dim),
            dtype=self.dtype,
        )
        value = jnp.zeros(
            (batch_size,) + (self.context_length, self.num_heads, head_dim),
            dtype=self.dtype,
        )
        return tuple(KVCache(position, mask, key, value) for _ in range(self.num_layers))

    def _add_positional_embedding(self, inputs, mask, carry: Carry):
        wpe = nn.Embed(
            num_embeddings=self.context_length,
            features=self.features,
            embedding_init=self.kernel_init,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="wpe",
        )
        kv_cache, *_ = carry
        position = kv_cache.position + jnp.cumsum(mask, axis=1, dtype=jnp.int32) % self.context_length

        jnp.isnan(position).any()
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
        initial_carry: Carry,
        **kwargs,
    ):
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
