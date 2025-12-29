from functools import partial
from typing import Any, Callable, Tuple, Optional

import jax
import jax.numpy as jnp
from jax import lax
from flax import linen as nn
from flax.linen import initializers
from flax.linen.module import compact

PRNGKey = Any
Shape = Tuple[int, ...]
Dtype = Any
Array = Any
Initializer = Callable[[PRNGKey, Shape, Dtype], Array]


def add_time_axis(x: jax.Array):
    return x[:, None, ...]


def broadcast_mask(mask: jax.Array, carry: jax.Array) -> jax.Array:
    while mask.ndim != carry.ndim:
        mask = mask[..., None] if mask.ndim < carry.ndim else mask[..., 0]
    return mask


class LinearAttentionLayer(nn.Module):
    head_dim: int
    num_heads: int
    kernel_init: Initializer
    bias_init: Initializer
    param_dtype: Dtype
    dtype: Optional[Dtype]

    @compact
    def __call__(
        self,
        inputs: Array,
        mask: Array,
        carry: Array,
    ) -> Tuple[Array, Array]:
        batch_size, sequence_length, in_features = inputs.shape
        hidden_dim = self.num_heads * self.head_dim

        projection = partial(
            nn.Dense,
            use_bias=False,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )

        q = projection(features=hidden_dim, name="query")(inputs).reshape(
            batch_size, sequence_length, self.num_heads, self.head_dim
        )
        k = projection(features=hidden_dim, name="key")(inputs).reshape(
            batch_size, sequence_length, self.num_heads, self.head_dim
        )
        v = projection(features=hidden_dim, name="value")(inputs).reshape(
            batch_size, sequence_length, self.num_heads, self.head_dim
        )

        q = nn.relu(q)
        k = nn.relu(k)

        B = jnp.matmul(v[..., None], k[..., None].swapaxes(-1, -2))

        A = jnp.broadcast_to(jnp.eye(self.head_dim), B.shape)
        A = A * (1.0 - broadcast_mask(mask, A))

        def binary_operator(lhs, rhs):
            a_i, b_i = lhs
            a_j, b_j = rhs
            return (jnp.matmul(a_i, a_j), jnp.matmul(b_i, a_j) + b_j)

        cummulative_decay, hidden_states = lax.associative_scan(
            jax.vmap(binary_operator), (A, B), axis=1
        )

        carry = jnp.matmul(add_time_axis(carry), cummulative_decay) + hidden_states

        x = (
            jnp.matmul(carry, q[..., None])
            .squeeze(-1)
            .reshape(batch_size, sequence_length, hidden_dim)
        )

        x = nn.RMSNorm(dtype=self.dtype)(x)

        y = nn.Dense(
            features=in_features, kernel_init=self.kernel_init, bias_init=self.bias_init
        )(x)

        final_carry = carry[:, -1]

        return final_carry, y


class SwiGLU(nn.Module):
    hidden_dim: int
    kernel_init: Initializer
    bias_init: Initializer
    param_dtype: Dtype
    dtype: Optional[Dtype]

    @compact
    def __call__(self, x):
        *_, in_features = x.shape

        dense = partial(
            nn.Dense,
            use_bias=False,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
            param_dtype=self.param_dtype,
            dtype=self.dtype,
        )

        gate = dense(features=self.hidden_dim, name="gate")(x)
        value = dense(features=self.hidden_dim, name="value")(x)

        x = nn.silu(gate) * value
        return dense(features=in_features, name="out")(x)


class LinearTransformerBlock(nn.Module):
    head_dim: int
    num_heads: int
    ratio: int = 4
    kernel_init: Initializer = initializers.lecun_normal()
    bias_init: Initializer = initializers.zeros_init()
    param_dtype: Dtype = jnp.float32
    dtype: Optional[Dtype] = None

    @compact
    def __call__(self, inputs: Array, mask: Array, carry: Array) -> Tuple[Array, Array]:

        *_, in_features = inputs.shape

        skip = inputs

        x = nn.RMSNorm(dtype=self.dtype)(inputs)

        carry, x = LinearAttentionLayer(
            head_dim=self.head_dim,
            num_heads=self.num_heads,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
            param_dtype=self.param_dtype,
            dtype=self.dtype,
        )(x, mask, carry)

        x = skip = x + skip

        x = nn.RMSNorm(dtype=self.dtype)(x)

        x = SwiGLU(
            hidden_dim=int(in_features * self.ratio),
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
            param_dtype=self.param_dtype,
            dtype=self.dtype,
        )(x)

        x = x + skip

        return carry, x

    def initialize_carry(
        self, rng: Optional[PRNGKey], input_shape: Tuple[int, ...]
    ) -> Array:
        batch_size, *_ = input_shape
        return jnp.zeros(
            (batch_size, self.num_heads, self.head_dim, self.head_dim), dtype=self.dtype
        )

    @property
    def num_feature_axes(self) -> int:
        return 1


class LinearTransformer(nn.Module):
    num_layers: int
    head_dim: int
    num_heads: int
    embedding_dim: int
    ratio: int = 4
    kernel_init: Initializer = initializers.lecun_normal()
    bias_init: Initializer = initializers.zeros_init()
    param_dtype: Dtype = jnp.float32
    dtype: Optional[Dtype] = None

    @compact
    def __call__(
        self,
        inputs: Array,
        mask: Array,
        initial_carry: Optional[Tuple[Array, ...]] = None,
        **kwargs,
    ) -> Tuple[Tuple[Array, ...], Array]:

        new_carry = []

        if initial_carry is None:
            initial_carry = self.initialize_carry(None, inputs.shape)

        for i, carry_i in enumerate(initial_carry):
            carry_i, inputs = LinearTransformerBlock(
                head_dim=self.head_dim,
                num_heads=self.num_heads,
                ratio=self.ratio,
                kernel_init=self.kernel_init,
                bias_init=self.bias_init,
                param_dtype=self.param_dtype,
                dtype=self.dtype,
                name=f"block_{i}",
            )(inputs, mask, carry_i)

            new_carry.append(carry_i)

        inputs = nn.RMSNorm(dtype=self.dtype)(inputs)

        inputs = nn.Dense(
            self.embedding_dim,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="head",
        )(inputs)

        return tuple(new_carry), inputs

    def initialize_carry(
        self, rng: Optional[PRNGKey], input_shape: Tuple[int, ...]
    ) -> Tuple[Array, ...]:
        batch_size, *_ = input_shape
        mem_shape = (batch_size, self.num_heads, self.head_dim, self.head_dim)

        return tuple(
            jnp.zeros(mem_shape, dtype=self.dtype or jnp.float32)
            for _ in range(self.num_layers)
        )
