from functools import partial
from typing import (
    Any,
    Optional,
    TypeVar,
)

import jax
from jax import numpy as jnp
from flax import linen as nn
from flax.linen import initializers
from flax import struct
from flax.linen.activation import sigmoid, tanh
from flax.linen.linear import Dense, default_kernel_init
from flax.linen.module import Module, compact, nowrap
from flax.typing import (
    Array,
    Dtype,
    Initializer,
)

from memorax.networks.sequence_models.sequence_model import SequenceModel

A = TypeVar("A")
Carry = Any


def _relative_shift(x: Array) -> Array:
    b, h, q_len, k_len = x.shape
    x = jnp.pad(x, ((0, 0), (0, 0), (0, 0), (1, 0)))
    x = x.reshape(b, h, k_len + 1, q_len)
    x = x[:, :, 1:, :]
    x = x.reshape(b, h, q_len, k_len)
    return x


def sinusoidal_positional_embedding(pos_seq: Array, dim: int) -> Array:
    inv_freq = 1.0 / (10000 ** (jnp.arange(0, dim, 2, dtype=jnp.float32) / dim))
    sinusoid = jnp.einsum("l,d->ld", pos_seq.astype(jnp.float32), inv_freq)
    positional_embedding = jnp.concatenate(
        [jnp.sin(sinusoid), jnp.cos(sinusoid)], axis=-1
    )
    return positional_embedding


def _build_positional_embedding(
    memory_length: int, context_length: int, sequence_length: int, dim: int
) -> Array:
    length = memory_length + context_length + sequence_length
    pos_seq = jnp.arange(length - 1, -1, -1)
    return sinusoidal_positional_embedding(pos_seq, dim)


@struct.dataclass
class KVCache:
    key: Array
    value: Array
    state: Array
    mask: Array


@struct.dataclass
class Memory:
    state: Array
    mask: Array


class RelativeMultiHeadAttentionBlock(Module):
    features: int
    num_heads: int
    context_length: int
    memory_length: int
    dtype: Dtype | None = None
    param_dtype: Dtype = jnp.float32
    kernel_init: Initializer = default_kernel_init
    bias_init: Initializer = initializers.zeros_init()
    dropout: float = 0.0

    @compact
    def __call__(
        self,
        x: Array,
        mask: Array,
        kv_cache: KVCache,
        memory: Memory,
        relative_positional_embeddings: Array,
    ):
        B, T, *_ = x.shape
        head_dim = self.features // self.num_heads

        assert (
            T <= self.context_length + self.memory_length
        ), f"T must be less than or equal to context_length + memory_length, but was T: {T}, context_length + memory_length: {self.context_length + self.memory_length}"

        projection = partial(
            nn.DenseGeneral,
            features=(self.num_heads, head_dim),
            use_bias=False,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            kernel_init=self.kernel_init,
        )

        query = projection(name="query")(x)

        key = jnp.concatenate([memory.state, x], axis=1)
        key = projection(name="key")(key)
        key = jnp.concatenate(
            [key[:, :-T, ...], kv_cache.key, key[:, -T:, ...]], axis=1
        )
        key = key[:, -(self.memory_length + self.context_length) :]

        value = jnp.concatenate([memory.state, x], axis=1)
        value = projection(name="value")(value)
        value = jnp.concatenate(
            [value[:, :-T, ...], kv_cache.value, value[:, -T:, ...]], axis=1
        )
        value = value[:, -(self.memory_length + self.context_length) :]

        r = projection(name="relative_positional_embeddings", axis=-1)(
            relative_positional_embeddings
        )

        u = self.param(
            "u", self.bias_init, (self.num_heads, head_dim), self.param_dtype
        )
        v = self.param(
            "v", self.bias_init, (self.num_heads, head_dim), self.param_dtype
        )
        query_u = query + u[None, None, :, :]
        query_v = query + v[None, None, :, :]

        bd = jnp.einsum("btnh,mnh->btnm", query_v, r)
        bd = jnp.transpose(bd, (0, 2, 1, 3))

        bd = _relative_shift(bd)[..., -(self.memory_length + self.context_length) :]

        bias = (bd / jnp.sqrt(head_dim)).astype(self.param_dtype)

        query_input = (
            jnp.cumsum(mask.astype(jnp.int32), axis=1)
            + jnp.max(
                jnp.cumsum(
                    jnp.concatenate([memory.mask, kv_cache.mask], axis=1), axis=1
                ),
                axis=1,
            )[..., None]
        )

        key_mask = jnp.concatenate(
            [memory.mask, kv_cache.mask, mask], axis=1, dtype=jnp.int32
        )
        key_input = jnp.cumsum(key_mask, axis=1)[
            :, -(self.memory_length + self.context_length) :
        ]

        attention_mask = nn.make_attention_mask(
            query_input, key_input, pairwise_fn=jnp.equal
        )

        query_input = jnp.arange(T) + self.memory_length + self.context_length
        query_input = jnp.broadcast_to(query_input, (B, T))
        key_input = jnp.arange(self.memory_length + self.context_length + T)
        key_input = jnp.broadcast_to(
            key_input, (B, self.memory_length + self.context_length + T)
        )[:, -(self.memory_length + self.context_length) :]
        causal_mask = nn.make_attention_mask(
            query_input, key_input, pairwise_fn=jnp.greater_equal
        )

        B, _, T, S = attention_mask.shape
        attention_mask = jnp.broadcast_to(attention_mask, (B, self.num_heads, T, S))

        B, _, T, S = causal_mask.shape
        causal_mask = jnp.broadcast_to(causal_mask, (B, self.num_heads, T, S))

        x = jax.nn.dot_product_attention(
            query_u.astype(jnp.bfloat16),
            key.astype(jnp.bfloat16),
            value.astype(jnp.bfloat16),
            mask=nn.combine_masks(attention_mask, causal_mask, dtype=jnp.bool),
            bias=bias,
            # implementation=get_attention_implementation(),
            implementation="xla",
        )

        x = nn.DenseGeneral(
            self.features,
            axis=(-2, -1),
            use_bias=False,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
            name="out",
        )(x)

        x = nn.Dropout(rate=self.dropout)(x, deterministic=not self.has_rng("dropout"))

        key = key[:, -self.context_length :]
        value = value[:, -self.context_length :]
        kv_cache = kv_cache.replace(key=key, value=value)
        return x, kv_cache


class GRUGate(Module):
    features: int
    gate_init_bias: float = 2.0
    kernel_init: Initializer = default_kernel_init
    bias_init: Initializer = initializers.zeros_init()
    dtype: Dtype | None = None
    param_dtype: Dtype = jnp.float32

    @compact
    def __call__(self, x: Array, y: Array) -> Array:
        dense = partial(
            Dense,
            features=self.features,
            use_bias=False,
            kernel_init=self.kernel_init,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )
        gate_bias = self.param(
            "gate_bias",
            nn.initializers.constant(self.gate_init_bias),
            (self.features,),
            self.param_dtype,
        )

        r = sigmoid(dense(name="r_y")(y) + dense(name="r_x")(x))
        z = sigmoid(dense(name="z_y")(y) + dense(name="z_x")(x) - gate_bias)
        h_tilde = tanh(dense(name="g_y")(y) + dense(name="g_x")(r * x))
        return (1.0 - z) * x + z * h_tilde


class MLP(Module):
    features: int
    hidden_dim: int
    dtype: Dtype | None = None
    param_dtype: Dtype = jnp.float32
    kernel_init: Initializer = default_kernel_init
    bias_init: Initializer = initializers.zeros_init()

    @compact
    def __call__(self, x: Array) -> Array:
        projection = partial(
            nn.Dense,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
        )
        x = projection(features=self.hidden_dim, name="up_proj")(x)
        x = nn.relu(x)
        x = projection(features=self.features, name="down_proj")(x)
        return x


class GTrXLBlock(nn.Module):
    features: int
    num_heads: int
    hidden_dim: int
    context_length: int
    memory_length: int
    dtype: Dtype | None
    param_dtype: Dtype
    kernel_init: Initializer
    bias_init: Initializer

    @compact
    def __call__(
        self,
        x: Array,
        mask: Array,
        kv_cache: KVCache,
        memory: Memory,
        relative_positional_embeddings: Array,
    ):
        gate = partial(
            GRUGate,
            features=self.features,
            kernel_init=self.kernel_init,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )
        layer_norm = partial(
            nn.LayerNorm, epsilon=1e-5, dtype=self.dtype, param_dtype=self.param_dtype
        )
        pre_norm = layer_norm(name="pre_norm")
        post_norm = layer_norm(name="post_norm")

        skip = x

        x = pre_norm(x)
        memory = memory.replace(state=pre_norm(memory.state))
        x, kv_cache = RelativeMultiHeadAttentionBlock(
            features=self.features,
            num_heads=self.num_heads,
            context_length=self.context_length,
            memory_length=self.memory_length,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            kernel_init=self.kernel_init,
        )(x, mask, kv_cache, memory, relative_positional_embeddings)
        x = nn.relu(x)
        x = gate(name="attn_gate")(skip, x)
        skip = x
        x = post_norm(x)
        x = MLP(
            features=self.features,
            hidden_dim=self.hidden_dim,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
        )(x)
        x = nn.relu(x)
        x = gate(name="output_gate")(skip, x)
        return x, kv_cache


class GTrXL(SequenceModel):
    features: int
    num_layers: int = 12
    num_heads: int = 12
    hidden_dim: Optional[int] = None
    context_length: int = 1024
    memory_length: int = 1024
    dtype: Dtype | None = None
    param_dtype: Dtype = jnp.float32
    kernel_init: Initializer = default_kernel_init
    bias_init: Initializer = initializers.zeros_init()

    @nowrap
    def initialize_carry(self, rng, input_shape):
        batch_size, *_ = input_shape

        def initialize_kv_cache():
            head_dim = self.features // self.num_heads
            key = jnp.zeros(
                (batch_size,) + (self.context_length, self.num_heads, head_dim),
                dtype=self.dtype,
            )
            value = jnp.zeros(
                (batch_size,) + (self.context_length, self.num_heads, head_dim),
                dtype=self.dtype,
            )
            state = jnp.zeros(
                (batch_size,) + (self.context_length, self.features),
                dtype=self.dtype,
            )
            mask = jnp.ones((batch_size, self.context_length), dtype=jnp.int32)
            return KVCache(key, value, state, mask)

        def initialize_memory():
            state = jnp.zeros(
                (batch_size,) + (self.memory_length, self.features),
                dtype=self.dtype,
            )
            mask = jnp.ones((batch_size, self.memory_length), dtype=jnp.int32)
            return Memory(state, mask)

        kv_cache = initialize_kv_cache()
        memory = initialize_memory()

        return tuple((kv_cache, memory) for _ in range(self.num_layers))

    @nn.compact
    def __call__(
        self,
        inputs: jax.Array,
        mask: jax.Array,
        initial_carry: tuple,
        **kwargs,
    ):

        new_carry = []
        x: Array = inputs
        carry: Carry = initial_carry

        _, T, *_ = x.shape
        relative_positional_embeddings = _build_positional_embedding(
            self.memory_length, self.context_length, T, self.features
        )

        for layer_idx, (kv_cache, memory) in enumerate(carry):
            x, kv_cache = GTrXLBlock(
                features=self.features,
                num_heads=self.num_heads,
                hidden_dim=self.hidden_dim or 4 * self.features,
                context_length=self.context_length,
                memory_length=self.memory_length,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                kernel_init=self.kernel_init,
                bias_init=self.bias_init,
                name=f"layer_{layer_idx}",
            )(x, mask, kv_cache, memory, relative_positional_embeddings)

            full_state = jnp.concatenate(
                [memory.state, kv_cache.state, jax.lax.stop_gradient(x)], axis=1
            )
            full_mask = jnp.concatenate([memory.mask, kv_cache.mask, mask], axis=1)

            kv_cache_state = full_state[:, -self.context_length :, :]
            kv_cache_mask = full_mask[:, -self.context_length :]
            kv_cache = kv_cache.replace(state=kv_cache_state, mask=kv_cache_mask)

            memory_state = full_state[
                :, -(self.memory_length + self.context_length) : -self.context_length, :
            ]
            memory_mask = full_mask[
                :, -(self.memory_length + self.context_length) : -self.context_length
            ]
            memory = memory.replace(state=memory_state, mask=memory_mask)

            new_carry.append((kv_cache, memory))
        new_carry = tuple(new_carry)

        return new_carry, x
