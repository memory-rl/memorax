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


class GRUGating(Module):
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


class RelativeMultiHeadAttentionBlock(Module):
    features: int
    num_heads: int
    head_dim: int | None = None
    dtype: Dtype | None = None
    param_dtype: Dtype = jnp.float32
    kernel_init: Initializer = default_kernel_init
    bias_init: Initializer = initializers.zeros_init()
    dropout: float = 0.0

    @compact
    def __call__(
        self,
        x: Array,
        relative_positional_embeddings: Array,
        memory: Array,
        episode_idx: Array,
        dones: Optional[Array] = None,
    ):
        B, T, *_ = x.shape
        _, M, *_ = memory.shape
        head_dim = self.head_dim or (self.features // self.num_heads)

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

        key = projection(name="key")(jnp.concatenate([memory, x], axis=1))
        value = projection(name="value")(jnp.concatenate([memory, x], axis=1))
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
        bd = _relative_shift(bd)[..., -(M+T):]

        bias = (bd / jnp.sqrt(head_dim)).astype(self.param_dtype)

        mask = None
        if dones is not None:
            query_input = episode_idx[:, -1:] + jnp.cumsum(dones, axis=1)
            key_input = jnp.concatenate([episode_idx, query_input], axis=1)
            attention_mask = nn.make_attention_mask(query_input, key_input, pairwise_fn=jnp.equal).astype(jnp.bool_)

            casual_mask = nn.make_causal_mask(jnp.ones((B, M + T)), dtype=jnp.bool_)[..., -T:, :]
            mask = nn.combine_masks(attention_mask, casual_mask).astype(jnp.bool_)

        x = jax.nn.dot_product_attention(
            query_u.astype(jnp.bfloat16),
            key.astype(jnp.bfloat16),
            value.astype(jnp.bfloat16),
            mask=mask,
            bias=bias,
            implementation="xla"
        ).reshape(B, T, self.num_heads * head_dim)


        x = nn.DenseGeneral(
            self.features,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
            name="out",
        )(x)

        x = nn.Dropout(rate=self.dropout)(x, deterministic=not self.has_rng("dropout"))
        return x


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


class GTrXLBlock(Module):
    features: int
    num_heads: int
    head_dim: int | None
    hidden_dim: int
    dtype: Dtype | None
    param_dtype: Dtype
    kernel_init: Initializer
    bias_init: Initializer

    @compact
    def __call__(
        self,
        x: Array,
        relative_positional_embedding: Array,
        memory: Array,
        episode_idx: Array,
        dones: Optional[Array] = None,
    ):
        gate = partial(
            GRUGating,
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
        memory = nn.LayerNorm(
            epsilon=1e-5,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )(memory)
        x = RelativeMultiHeadAttentionBlock(
            features=self.features,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            kernel_init=self.kernel_init,
        )(x, relative_positional_embedding, memory, episode_idx, dones)
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


class GTrXLCell(RNNCellBase):
    features: int
    num_layers: int
    num_heads: int
    head_dim: int | None = None
    hidden_dim: int | None = None
    memory_length: int = 1024
    dtype: Dtype | None = None
    param_dtype: Dtype = jnp.float32
    carry_dtype: Dtype = jnp.float32
    kernel_init: Initializer = default_kernel_init
    bias_init: Initializer = initializers.zeros_init()
    carry_init: Initializer = initializers.zeros_init()

    @property
    def num_feature_axes(self) -> int:
        return 1

    def _build_positional_embedding(
        self, memory_length: int, sequence_length: int, dim: int
    ) -> Array:
        length = memory_length + sequence_length
        pos_seq = jnp.arange(length - 1, -1, -1)
        return sinusoidal_positional_embedding(pos_seq, dim)

    @compact
    def __call__(
        self, carry: tuple[Any, ...], x: Array
    ) -> tuple[tuple[Any, ...], Array]:
        head_dim = self.head_dim or (self.features // self.num_heads)

        x = x[:, None, :]
        _, sequence_length, _ = x.shape
        relative_positional_embedding = self._build_positional_embedding(
            self.memory_length, sequence_length, self.features
        )

        new_carry = []
        for layer_idx, (memory, episode_idx) in zip(range(self.num_layers), carry):
            x = GTrXLBlock(
                features=self.features,
                num_heads=self.num_heads,
                head_dim=head_dim,
                hidden_dim=self.hidden_dim or 4 * self.features,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                kernel_init=self.kernel_init,
                bias_init=self.bias_init,
                name=f"layer_{layer_idx}",
            )(x, relative_positional_embedding, memory, episode_idx)
            memory = jnp.concatenate(
                [memory, jax.lax.stop_gradient(x)], axis=1
            )[:, -self.memory_length :, :].astype(self.carry_dtype)


            new_carry.append((memory, episode_idx))
        new_carry = tuple(new_carry)

        x = x.squeeze(axis=1)
        return new_carry, x

    @compact
    def apply_parallel(
        self, carry: tuple[Any, ...], x: Array, dones: Array
    ) -> tuple[tuple[Any, ...], Array]:
        head_dim = self.head_dim or (self.features // self.num_heads)

        _, sequence_length, _ = x.shape
        relative_positional_embedding = self._build_positional_embedding(
            self.memory_length, sequence_length, self.features
        )

        new_carry = []
        for layer_idx, (memory, episode_idx) in zip(range(self.num_layers), carry):
            x = GTrXLBlock(
                features=self.features,
                num_heads=self.num_heads,
                head_dim=head_dim,
                hidden_dim=self.hidden_dim or 4 * self.features,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                kernel_init=self.kernel_init,
                bias_init=self.bias_init,
                name=f"layer_{layer_idx}",
            )(x, relative_positional_embedding, memory, episode_idx, dones)
            memory = jnp.concatenate(
                [memory, jax.lax.stop_gradient(x)], axis=1
            )[:, -self.memory_length :, :].astype(self.carry_dtype)
            query_input = episode_idx[:, -1:] + jnp.cumsum(dones, axis=1)
            episode_idx = jnp.concatenate([episode_idx, query_input], axis=1)[:, -self.memory_length:]
            new_carry.append((memory, episode_idx))
        new_carry = tuple(new_carry)

        return new_carry, x

    @nowrap
    def initialize_carry(self, rng, input_shape):
        batch_size, *_ = input_shape

        carry = tuple(
            (self.carry_init(
                random.fold_in(rng, i),
                ((batch_size,) + (self.memory_length, self.features)),
                self.carry_dtype,
            ), jnp.ones((batch_size, self.memory_length), dtype=jnp.int32))
            for i in range(self.num_layers)
        )
        return carry
