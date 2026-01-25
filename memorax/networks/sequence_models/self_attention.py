from functools import partial
from typing import Any, Optional

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax import struct

from memorax.networks.positional_embeddings import RelativePositionalEmbedding
from memorax.networks.sequence_models.utils import (
    get_attention_implementation, get_attention_mask, get_input_shape)
from memorax.utils.typing import Array

from .sequence_model import SequenceModel


@struct.dataclass
class Carry:
    mask: Array
    key: Array
    value: Array


class SelfAttention(SequenceModel):
    features: int
    num_heads: int
    context_length: int
    dtype: Any
    param_dtype: Any
    kernel_init: Any
    bias_init: Any
    positional_embedding: RelativePositionalEmbedding = (
        lambda query, key, query_pos, key_pos: (
            query,
            key,
            None,
        )
    )

    @nn.nowrap
    def initialize_carry(self, key, input_shape):
        *batch_dims, _ = input_shape
        head_dim = self.features // self.num_heads
        mask = jnp.ones((*batch_dims, self.context_length), dtype=jnp.int32)
        key = jnp.zeros(
            (*batch_dims, self.context_length, self.num_heads, head_dim),
            dtype=self.dtype,
        )
        value = jnp.zeros(
            (*batch_dims, self.context_length, self.num_heads, head_dim),
            dtype=self.dtype,
        )
        return Carry(mask, key, value)

    @nn.compact
    def __call__(
        self,
        x,
        mask,
        initial_carry: Optional[Carry] = None,
        memory: Optional[Array] = None,
        memory_mask: Optional[Array] = None,
        **kwargs,
    ):
        if initial_carry is None:
            input_shape = get_input_shape(x)
            initial_carry = self.initialize_carry(jax.random.key(0), input_shape)

        B, T, *_ = x.shape
        head_dim = self.features // self.num_heads

        if memory is None:
            memory = jnp.zeros((B, 0, self.features), dtype=self.dtype)
            memory_mask = jnp.zeros((B, 0), dtype=jnp.int32)

        _, M, *_ = memory.shape

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

        key = projection(name="key")(jnp.concatenate([memory, x], axis=1))
        key = jnp.concatenate([key[:, :M], initial_carry.key, key[:, M:]], axis=1)
        key = key[:, -(M + self.context_length) :]

        value = projection(name="value")(jnp.concatenate([memory, x], axis=1))
        value = jnp.concatenate(
            [value[:, :M], initial_carry.value, value[:, M:]], axis=1
        )
        value = value[:, -(M + self.context_length) :]

        attention_mask, query_input, key_input = get_attention_mask(
            mask, initial_carry, memory_mask, self.context_length, self.num_heads
        )

        query, key, bias = self.positional_embedding(query, key, query_input, key_input)

        implementation, attention_dtype = get_attention_implementation()
        x = jax.nn.dot_product_attention(
            query.astype(attention_dtype),
            key.astype(attention_dtype),
            value.astype(attention_dtype),
            bias=bias,
            mask=attention_mask,
            implementation=implementation,
        ).astype(self.dtype)

        y = nn.DenseGeneral(
            self.features,
            axis=(-2, -1),
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
            name="out",
        )(x)

        mask = jnp.concatenate([initial_carry.mask, mask], axis=1)[
            :, -self.context_length :
        ]
        key = key[:, -self.context_length :]
        value = value[:, -self.context_length :]
        carry = initial_carry.replace(mask=mask, key=key, value=value)

        return carry, y
