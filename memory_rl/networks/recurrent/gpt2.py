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
    def read(kv_cache, context_length):
        length = KVCache.length(kv_cache.idx, context_length)

        start_idx = (kv_cache.idx - length) % context_length
        positions = (jnp.arange(context_length, dtype=jnp.int32)[None, :] +
                 start_idx[:, None]) % context_length
        k = jnp.take_along_axis(kv_cache.key,   positions[:, :, None, None], axis=1)
        v = jnp.take_along_axis(kv_cache.value, positions[:, :, None, None], axis=1)
        return k, v, length

    @staticmethod
    def update(kv_cache, context_length, key, value):
        idx = kv_cache.idx % context_length

        key = jax.lax.dynamic_update_slice(kv_cache.key, key, (idx, 0, 0))
        value = jax.lax.dynamic_update_slice(kv_cache.value, value, (idx, 0, 0))

        return KVCache(idx + 1, key, value)

    @staticmethod
    def length(idx, context_length):
        return jnp.minimum(idx, context_length)


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
    def __call__(self, x, kv_cache=None, dones=None):
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


        if kv_cache is not None:
            def _add_time_axis(x):
                return x[:, None, :, :]

            query = _add_time_axis(query)
            key = _add_time_axis(key)
            value = _add_time_axis(value)

            kv_cache = jax.vmap(KVCache.update, in_axes=(0, None, 0, 0))(
                kv_cache, self.context_length, key, value
            )

            key, value, key_value_seq_lengths = KVCache.read(kv_cache, self.context_length)

            x = jax.nn.dot_product_attention(
                query.astype(jnp.bfloat16),
                key.astype(jnp.bfloat16),
                value.astype(jnp.bfloat16),
                key_value_seq_lengths=key_value_seq_lengths,
                implementation="cudnn"
            ).squeeze(axis=1)
        elif dones is not None:
            episode_idx = jnp.cumsum(dones, axis=1)
            mask = nn.make_attention_mask(episode_idx, episode_idx, pairwise_fn=jnp.equal).astype(jnp.bool_)
            B, _, T, S = mask.shape
            mask = jnp.broadcast_to(mask, (B, self.num_heads, T, S))

            x = jax.nn.dot_product_attention(
                query.astype(jnp.bfloat16),
                key.astype(jnp.bfloat16),
                value.astype(jnp.bfloat16),
                is_causal=True,
                mask=mask,
                implementation="cudnn"
            )
        else:
            raise ValueError("Either kv_cache or dones must be provided")

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
    def __call__(self, x, kv_cache=None, dones=None):
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
        )(x, kv_cache, dones)
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
    context_length: int = 1024
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
            (batch_size,) + (self.context_length, self.num_heads, head_dim),
            dtype=self.dtype,
        )
        value = jnp.zeros(
            (batch_size,) + (self.context_length, self.num_heads, head_dim),
            dtype=self.dtype,
        )
        return tuple(KVCache(idx, key, value) for _ in range(self.num_layers))


    @nn.compact
    def __call__(self, carry, x):

        def _add_positional_embedding(carry, x):
            kv_cache, *_ = carry
            wpe = nn.Embed(
                num_embeddings=self.context_length,
                features=self.features,
                embedding_init=self.kernel_init,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                name="wpe",
            )

            idx = kv_cache.idx % self.context_length
            x = x + wpe(idx)
            x = nn.Dropout(rate=self.dropout)(x, deterministic=not self.has_rng("dropout"))
            return x

        x = _add_positional_embedding(carry, x)

        new_carry = []
        for i, kv_cache in zip(range(self.num_layers), carry):
            x, kv_cache = GPT2Block(
                features=self.features,
                num_heads=self.num_heads,
                hidden_dim=self.hidden_dim or 4 * self.features,
                context_length=self.context_length,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                kernel_init=self.kernel_init,
                bias_init=self.bias_init,
                name=f"layer_{i}",
            )(x, kv_cache=kv_cache)
            new_carry.append(kv_cache)
        new_carry = tuple(new_carry)

        x = nn.LayerNorm(
            epsilon=1e-5,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )(x)
        return new_carry, x

    @nn.compact
    def apply_parallel(self, carry, x, dones):

        def _add_positional_embedding(x, dones):
            _, T = dones.shape
            episode_starts = jnp.pad(dones[:, :-1], ((0, 0), (1, 0)))
            t = jnp.arange(T)[None, :]
            last_start = jnp.maximum.accumulate(jnp.where(episode_starts, t, 0), axis=1)
            pos = (t - last_start) % self.context_length
            wpe = nn.Embed(
                num_embeddings=self.context_length,
                features=self.features,
                embedding_init=self.kernel_init,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                name="wpe",
            )
            x = x + wpe(pos)
            x = nn.Dropout(rate=self.dropout)(x, deterministic=not self.has_rng("dropout"))
            return x

        x = _add_positional_embedding(x, dones)
        for i in range(self.num_layers):
            x, _ = GPT2Block(
                features=self.features,
                num_heads=self.num_heads,
                hidden_dim=self.hidden_dim or 4 * self.features,
                context_length=self.context_length,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                kernel_init=self.kernel_init,
                bias_init=self.bias_init,
                name=f"layer_{i}",
            )(x, dones=dones)

        x = nn.LayerNorm(
            epsilon=1e-5,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )(x)
        return carry, x
