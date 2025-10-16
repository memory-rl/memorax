from functools import partial
from typing import (
    Any,
    Optional,
    TypeVar,
)

import jax
from jax import numpy as jnp
from jax import random

from flax import linen as nn
from flax.linen import initializers, LayerNorm
from flax import struct
from flax.linen.activation import sigmoid, tanh, softmax
from flax.linen.linear import Dense, default_kernel_init
from flax.linen.module import Module, compact, nowrap
from flax.linen.recurrent import RNNCellBase
from flax.typing import (
    Array,
    PRNGKey,
    Dtype,
    Initializer,
)

from memory_rl.networks.recurrent.utils import get_attention_implementation

A = TypeVar("A")
Carry = Any
CarryHistory = Any
Output = Any


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
    memory_length: int, sequence_length: int, dim: int
) -> Array:
    length = memory_length + sequence_length
    pos_seq = jnp.arange(length - 1, -1, -1)
    return sinusoidal_positional_embedding(pos_seq, dim)


@struct.dataclass
class Memory:
    position: Array
    mask: Array
    state: Array


class RelativeMultiHeadAttentionBlock(Module):
    features: int
    num_heads: int
    context_length: int
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
        memory: Memory,
    ):
        B, T, *_ = x.shape
        head_dim = self.features // self.num_heads

        assert (
            self.features == self.num_heads * head_dim
        ), "d_model must equal n_heads * d_head"

        projection = partial(
            nn.DenseGeneral,
            features=(self.num_heads, head_dim),
            use_bias=False,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            kernel_init=self.kernel_init,
        )

        query = projection(name="query")(x)

        key = projection(name="key")(jnp.concatenate([memory.state, x], axis=1))
        value = projection(name="value")(jnp.concatenate([memory.state, x], axis=1))

        relative_positional_embeddings = _build_positional_embedding(
            self.context_length, T, self.features
        )
        r = projection(name="relative_positional_embeddings")(
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
        bd = _relative_shift(bd)[..., -(self.context_length + T) :]

        bias = (bd / jnp.sqrt(head_dim)).astype(self.param_dtype)
        bias = jax.lax.stop_gradient(bias)

        query_mask = mask.astype(jnp.int32)
        query_input = jax.lax.cumsum(query_mask, reverse=True, axis=1)

        key_mask = jnp.concatenate([memory.mask, mask], axis=1, dtype=jnp.int32)
        key_input = jax.lax.cumsum(key_mask, reverse=True, axis=1)

        attention_mask = nn.make_attention_mask(
            query_input, key_input, pairwise_fn=jnp.equal
        ).astype(jnp.bool_)

        B, _, T, S = attention_mask.shape
        attention_mask = jnp.broadcast_to(attention_mask, (B, self.num_heads, T, S))

        x = jax.nn.dot_product_attention(
            query_u.astype(jnp.bfloat16),
            key.astype(jnp.bfloat16),
            value.astype(jnp.bfloat16),
            is_causal=True,
            mask=attention_mask,
            bias=bias,
            # implementation=get_attention_implementation(),
            implementation="xla",
        )

        x = nn.DenseGeneral(
            self.features,
            axis=(-2, -1),
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
            name="out",
        )(x)

        x = nn.Dropout(rate=self.dropout)(x, deterministic=not self.has_rng("dropout"))
        return x


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
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )
        r = sigmoid(
            dense(use_bias=True, name="r_y")(y) + dense(use_bias=False, name="r_x")(x)
        )
        z = sigmoid(
            dense(use_bias=True, name="z_y")(y)
            + dense(use_bias=False, name="z_x")(x)
            - self.gate_init_bias
        )
        h_tilde = tanh(
            dense(use_bias=True, name="h_y")(y)
            + dense(use_bias=False, name="h_x")(r * x)
        )
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
        x = jax.nn.relu(x)
        x = projection(features=self.features, name="down_proj")(x)
        return x


class GTrXLBlock(nn.Module):
    features: int
    num_heads: int
    hidden_dim: int
    context_length: int
    dtype: Dtype | None
    param_dtype: Dtype
    kernel_init: Initializer
    bias_init: Initializer

    @compact
    def __call__(
        self,
        x: Array,
        mask: Array,
        memory: Memory,
    ):
        gate = partial(
            GRUGate,
            features=self.features,
            kernel_init=self.kernel_init,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )
        skip = x
        x = nn.LayerNorm(
            epsilon=1e-5,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )(x)
        state = nn.LayerNorm(
            epsilon=1e-5,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )(memory.state)
        memory = memory.replace(state=state)
        x = RelativeMultiHeadAttentionBlock(
            features=self.features,
            num_heads=self.num_heads,
            context_length=self.context_length,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            kernel_init=self.kernel_init,
        )(x, mask, memory)
        x = gate(name="attn_gate")(skip, x)
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
        x = gate(name="output_gate")(skip, x)
        return x


class GTrXL(RNNCellBase):
    features: int
    num_layers: int = 12
    num_heads: int = 12
    hidden_dim: Optional[int] = None
    context_length: int = 1024
    dtype: Dtype | None = None
    param_dtype: Dtype = jnp.float32
    kernel_init: Initializer = default_kernel_init
    bias_init: Initializer = initializers.zeros_init()

    @property
    def num_feature_axes(self) -> int:
        return 1

    @nowrap
    def initialize_carry(self, rng, input_shape):
        batch_size, *_ = input_shape

        position = jnp.full((batch_size, self.context_length), -1, dtype=jnp.int32)
        mask = jnp.ones((batch_size, self.context_length), dtype=jnp.int32)
        state = jnp.zeros(
            (batch_size,) + (self.context_length, self.features),
            dtype=self.dtype,
        )

        return tuple(Memory(position, mask, state) for _ in range(self.num_layers))

    @nn.compact
    def __call__(
        self,
        inputs: jax.Array,
        mask: jax.Array,
        initial_carry: Memory,
    ):

        new_carry = []
        x: Array = inputs
        carry: Carry = initial_carry
        for layer_idx, memory in enumerate(carry):
            x = GTrXLBlock(
                features=self.features,
                num_heads=self.num_heads,
                hidden_dim=self.hidden_dim or 4 * self.features,
                context_length=self.context_length,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                kernel_init=self.kernel_init,
                bias_init=self.bias_init,
                name=f"layer_{layer_idx}",
            )(x, mask, memory)
            memory_state = jnp.concatenate(
                [memory.state, jax.lax.stop_gradient(x)], axis=1
            )
            memory_state = memory_state[:, -self.context_length :, :]

            memory_mask = jnp.concatenate([memory.mask, mask], axis=1, dtype=jnp.int32)
            memory_mask = memory_mask[:, -self.context_length :]

            memory = memory.replace(state=memory_state, mask=memory_mask)
            new_carry.append(memory)
        new_carry = tuple(new_carry)

        return new_carry, x
