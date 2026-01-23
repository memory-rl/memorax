from functools import partial
from typing import Any, Optional

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax import struct

from memorax.networks.sequence_models.utils import (
    get_attention_implementation,
    get_input_shape,
)
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
    dropout: float = 0.0

    @nn.nowrap
    def initialize_carry(self, key, input_shape):
        batch_size, *_ = input_shape
        head_dim = self.features // self.num_heads
        mask = jnp.ones((batch_size, self.context_length), dtype=jnp.int32)
        key = jnp.zeros(
            (batch_size,) + (self.context_length, self.num_heads, head_dim),
            dtype=self.dtype,
        )
        value = jnp.zeros(
            (batch_size,) + (self.context_length, self.num_heads, head_dim),
            dtype=self.dtype,
        )
        return Carry(mask, key, value)

    @nn.compact
    def __call__(self, x, mask, initial_carry: Optional[Carry] = None, **kwargs):

        if initial_carry is None:
            input_shape = get_input_shape(x)
            initial_carry = self.initialize_carry(jax.random.key(0), input_shape)

        head_dim = self.features // self.num_heads

        B, T, *_ = x.shape

        assert (
            T <= self.context_length
        ), f"T must be less than or equal to context_length, but was T: {T}, context_length: {self.context_length}"

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
        key = jnp.concatenate([initial_carry.key, key], axis=1)
        key = key[:, -self.context_length :]

        value = projection(name="value")(x)
        value = jnp.concatenate([initial_carry.value, value], axis=1)
        value = value[:, -self.context_length :]

        query_input = (
            jnp.cumsum(mask.astype(jnp.int32), axis=1)
            + jnp.max(jnp.cumsum(initial_carry.mask, axis=1), axis=1)[..., None]
        )

        key_mask = jnp.concatenate([initial_carry.mask, mask], axis=1, dtype=jnp.int32)
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

        mask = key_mask[:, -self.context_length :]
        initial_carry = initial_carry.replace(mask=mask, key=key, value=value)

        return y, initial_carry
