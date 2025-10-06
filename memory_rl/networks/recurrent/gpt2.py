from typing import Any, Optional
from functools import partial
from flax.linen.module import nowrap
from flax.typing import Dtype
import jax.numpy as jnp
import jax
from flax import linen as nn
from flax import struct
from flax.linen.recurrent import RNNCellBase

@struct.dataclass
class KVCache:
    idx: jnp.ndarray
    key: jnp.ndarray
    value: jnp.ndarray

    @staticmethod
    def update(kv_cache, max_length, key, value):
        idx = kv_cache.idx % max_length

        key = jax.lax.dynamic_update_slice(kv_cache.key, key, (idx, 0, 0))
        value = jax.lax.dynamic_update_slice(kv_cache.value, value, (idx, 0, 0))

        return KVCache(idx + 1, key, value)

    @staticmethod
    def length(idx, max_length):
        return jnp.minimum(idx, max_length)


class MultiHeadAttentionBlock(nn.Module):
    features: int
    num_heads: int
    max_length: int
    dtype: Any
    param_dtype: Any
    kernel_init: Any
    bias_init: Any
    dropout: float = 0.0

    @nn.compact
    def __call__(self, x, kv_cache):
        head_dim = self.features // self.num_heads

        projection = partial(
            nn.DenseGeneral,
            features=(self.num_heads, head_dim),
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
        )

        def _add_time_axis(x):
            return x[:, None, :, :]

        query = _add_time_axis(projection(name="query")(x))
        key = _add_time_axis(projection(name="key")(x))
        value = _add_time_axis(projection(name="value")(x))

        kv_cache = jax.vmap(KVCache.update, in_axes=(0, None, 0, 0))(
            kv_cache, self.max_length, key, value
        )
        key_value_seq_lengths = KVCache.length(kv_cache.idx, self.max_length)

        x = jax.nn.dot_product_attention(
            query.astype(jnp.bfloat16),
            kv_cache.key.astype(jnp.bfloat16),
            kv_cache.value.astype(jnp.bfloat16),
            key_value_seq_lengths=key_value_seq_lengths,
            implementation="xla"
        )

        y = nn.DenseGeneral(
            self.features,
            axis=(-2, -1),
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
            name="out",
        )(x).squeeze(axis=1)

        y = nn.Dropout(rate=self.dropout)(y, deterministic=not self.has_rng("dropout"))
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
    max_length: int
    dtype: Any
    param_dtype: Any
    kernel_init: Any
    bias_init: Any

    @nn.compact
    def __call__(self, x, kv_cache):
        skip = x
        x = nn.LayerNorm(
            epsilon=1e-5,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )(x)
        x, kv_cache = MultiHeadAttentionBlock(
            features=self.features,
            num_heads=self.num_heads,
            max_length=self.max_length,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
        )(x, kv_cache)
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



class GPT2Cell(RNNCellBase):
    """GPT-2 as a Flax Linen RNNCell with a per-layer KV ring buffer.

    Carry: (k_cache, v_cache, t)
      - k_cache, v_cache: [B, n_layer, T, H, Hd]
      - t:                 [B] step counter (monotonic)
    """
    features: int
    num_layers: int = 12
    num_heads: int = 12
    hidden_dim: Optional[int] = None
    max_length: int = 1024
    dropout: float = 0.0
    dtype: Dtype | None = None
    param_dtype: Dtype = jnp.float32
    kernel_init: Any = nn.initializers.normal(stddev=0.02)
    bias_init: Any = nn.initializers.zeros_init()
    carry_init: Any = None  # unused; present for API parity

    @property
    def num_feature_axes(self) -> int:
        return 1

    @nowrap
    def initialize_carry(self, rng, input_shape):
        batch_size, *_ = input_shape
        head_dim = self.features // self.num_heads
        idx = jnp.zeros(batch_size, dtype=jnp.int32)
        key = jnp.zeros(
            (batch_size,) + (self.max_length, self.num_heads, head_dim),
            dtype=self.dtype,
        )
        value = jnp.zeros(
            (batch_size,) + (self.max_length, self.num_heads, head_dim),
            dtype=self.dtype,
        )
        return tuple(KVCache(idx, key, value) for _ in range(self.num_layers))

    def _add_positional_embedding(self, carry, x):
        kv_cache, *_ = carry
        wpe = nn.Embed(
            num_embeddings=self.max_length,
            features=self.features,
            embedding_init=self.kernel_init,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="wpe",
        )
        idx = kv_cache.idx % self.max_length
        x = x + wpe(idx)
        x = nn.Dropout(rate=self.dropout)(x, deterministic=not self.has_rng("dropout"))
        return x

    @nn.compact
    def __call__(self, carry, x):
        x = self._add_positional_embedding(carry, x)
        new_carry = []
        for i, kv_cache in zip(range(self.num_layers), carry):
            x, kv_cache = GPT2Block(
                features=self.features,
                num_heads=self.num_heads,
                hidden_dim=self.hidden_dim or 4 * self.features,
                max_length=self.max_length,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                kernel_init=self.kernel_init,
                bias_init=self.bias_init,
                name=f"layer_{i}",
            )(x, kv_cache)
            new_carry.append(kv_cache)
        new_carry = tuple(new_carry)

        x = nn.LayerNorm(
            epsilon=1e-5,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )(x)
        return new_carry, x
