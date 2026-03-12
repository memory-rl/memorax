from functools import partial
from typing import Literal, Optional, Tuple

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax import struct
from flax.typing import Dtype, Initializer

from memorax.networks.positional_embeddings import RelativePositionalEmbedding
from memorax.utils.axes import get_input_shape
from memorax.utils.typing import Array

Implementation = Literal["xla", "cudnn"]


def get_attention_implementation() -> tuple[Implementation, jnp.dtype]:
    backend = jax.default_backend()
    if backend == "gpu":
        try:
            if any(
                "nvidia" in device.device_kind.lower() for device in jax.local_devices()
            ):
                return "cudnn", jnp.bfloat16
        except Exception:
            pass
    return "xla", jnp.float32


def get_attention_mask(mask, initial_carry, memory_mask, context_length, num_heads):
    """Compute attention mask with position information for positional embeddings.

    Returns:
        Tuple of (combined_mask, query_input, key_input) where:
            - combined_mask: combined attention and causal mask (B, NH, T, S)
            - query_input: query positions (B, T)
            - key_input: key positions (B, M + context_length)
    """
    B, T = mask.shape
    _, M, *_ = memory_mask.shape

    query_mask = (
        jnp.cumsum(mask.astype(jnp.int32), axis=1)
        + jnp.max(
            jnp.cumsum(
                jnp.concatenate([memory_mask, initial_carry.mask], axis=1), axis=1
            ),
            axis=1,
        )[..., None]
    )

    key_mask = jnp.concatenate(
        [memory_mask, initial_carry.mask, mask], axis=1, dtype=jnp.int32
    )
    key_mask = jnp.cumsum(key_mask, axis=1)
    key_mask = key_mask[:, -(M + context_length) :]

    attention_mask = nn.make_attention_mask(query_mask, key_mask, pairwise_fn=jnp.equal)

    query_input = jnp.arange(T) + M + context_length
    query_input = jnp.broadcast_to(query_input, (B, T))
    key_input = jnp.arange(M + context_length + T)
    key_input = jnp.broadcast_to(key_input, (B, M + context_length + T))
    key_input = key_input[:, -(M + context_length) :]
    causal_mask = nn.make_attention_mask(
        query_input, key_input, pairwise_fn=jnp.greater_equal
    )

    B, _, T, S = attention_mask.shape
    attention_mask = jnp.broadcast_to(attention_mask, (B, num_heads, T, S))

    B, _, T, S = causal_mask.shape
    causal_mask = jnp.broadcast_to(causal_mask, (B, num_heads, T, S))

    combined_mask = nn.combine_masks(attention_mask, causal_mask, dtype=jnp.bool)
    return combined_mask, query_input, key_input

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
    dtype: Dtype = jnp.float32
    param_dtype: Dtype = jnp.float32
    kernel_init: Initializer = nn.initializers.lecun_normal()
    bias_init: Initializer = nn.initializers.zeros_init()
    positional_embedding: RelativePositionalEmbedding = (
        lambda query, key, query_pos, key_pos: (
            query,
            key,
            None,
        )
    )

    def setup(self):
        head_dim = self.features // self.num_heads

        projection = partial(
            nn.DenseGeneral,
            features=(self.num_heads, head_dim),
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
        )

        self.query = projection()
        self.key = projection()
        self.value = projection()
        self.output_projection = nn.DenseGeneral(
            self.features,
            axis=(-2, -1),
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
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

    def __call__(
        self,
        x,
        mask,
        initial_carry: Optional[Carry] = None,
        memory: Optional[Array] = None,
        memory_mask: Optional[Array] = None,
        **kwargs,
    ) -> Tuple[Carry, Array]:
        if initial_carry is None:
            input_shape = get_input_shape(x)
            initial_carry = self.initialize_carry(jax.random.key(0), input_shape)

        B, T, *_ = x.shape

        if memory is None:
            memory = jnp.zeros((B, 0, self.features), dtype=self.dtype)
            memory_mask = jnp.zeros((B, 0), dtype=jnp.int32)

        _, M, *_ = memory.shape

        assert (
            T <= self.context_length
        ), f"T must be less than or equal to context_length, but was T: {T}, context_length: {self.context_length}"

        query = self.query(x)

        key = self.key(jnp.concatenate([memory, x], axis=1))
        key = jnp.concatenate([key[:, :M], initial_carry.key, key[:, M:]], axis=1)
        key = key[:, -(M + self.context_length) :]

        value = self.value(jnp.concatenate([memory, x], axis=1))
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

        y = self.output_projection(x)

        mask = jnp.concatenate([initial_carry.mask, mask], axis=1)[
            :, -self.context_length :
        ]
        key = key[:, -self.context_length :]
        value = value[:, -self.context_length :]
        carry = initial_carry.replace(mask=mask, key=key, value=value)

        return carry, y
