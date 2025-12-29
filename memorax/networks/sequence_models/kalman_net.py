from functools import partial
from typing import Any, Callable, Optional, Tuple

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.linen import initializers
from flax.linen.module import compact
from jax import lax

PRNGKey = Any
Shape = Tuple[int, ...]
Dtype = Any
Array = Any
Initializer = Callable[[PRNGKey, Shape, Dtype], Array]

from memorax.networks.sequence_models.sequence_model import SequenceModel

def add_time_axis(x: jax.Array):
    return x[:, None, ...]


def broadcast_mask(mask: jax.Array, carry: jax.Array) -> jax.Array:
    while mask.ndim != carry.ndim:
        mask = mask[..., None] if mask.ndim < carry.ndim else mask[..., 0]
    return mask


class KalmanNetLayer(nn.Module):
    head_dim: int
    num_heads: int
    num_actions: int
    kernel_init: Initializer
    bias_init: Initializer
    param_dtype: Dtype
    dtype: Optional[Dtype]

    @compact
    def __call__(
        self,
        x: Array,
        action: Array,
        mask: Array,
        carry: Array,
    ) -> Tuple[Array, Array]:

        batch_size, sequence_length, in_features = x.shape
        hidden_dim = self.num_heads * self.head_dim

        projection = partial(
            nn.Dense,
            use_bias=False,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )
        q_s = projection(features=hidden_dim)(x).reshape(
            batch_size, sequence_length, self.num_heads, self.head_dim
        )
        k_s = projection(features=hidden_dim)(x).reshape(
            batch_size, sequence_length, self.num_heads, self.head_dim
        )
        v_s = projection(features=hidden_dim)(x).reshape(
            batch_size, sequence_length, self.num_heads, self.head_dim
        )
        beta = projection(features=self.num_heads)(x).reshape(
            batch_size, sequence_length, self.num_heads
        )

        # u = nn.Embed(
        #     num_embeddings=self.num_actions,
        #     features=hidden_dim,
        # )(action)
        # u = x
        #
        # alpha = projection(
        #     features=self.num_heads, bias_init=initializers.constant(4.0), name="alpha"
        # )(u).reshape(batch_size, sequence_length, self.num_heads)
        # gamma = projection(features=self.num_heads)(u).reshape(
        #     batch_size, sequence_length, self.num_heads
        # )
        # k_u = projection(features=hidden_dim, name="b_u")(u).reshape(
        #     batch_size, sequence_length, self.num_heads, self.head_dim
        # )
        # v_u = projection(features=hidden_dim, name="b_v")(u).reshape(
        #     batch_size, sequence_length, self.num_heads, self.head_dim
        # )

        q_s = nn.silu(q_s)
        k_s = nn.silu(k_s)
        k_u = nn.silu(k_u)

        def l2(x, axis=-1, keepdims=True, eps=1e-6):
            sum = jnp.sum(x * x, axis=axis, keepdims=keepdims)
            return jnp.sqrt(sum + eps)

        q_s = q_s / l2(q_s, axis=-1, keepdims=True)
        k_s = k_s / l2(k_s, axis=-1, keepdims=True)
        k_u = k_u / l2(k_u, axis=-1, keepdims=True)

        beta = nn.sigmoid(beta)
        A = nn.sigmoid(alpha)
        B = nn.sigmoid(gamma)

        delta = jnp.eye(self.head_dim, dtype=self.dtype) - (
            beta[..., None, None]
            * jnp.matmul(k_s[..., None], k_s[..., None].swapaxes(-1, -2))
        )
        decay_matrix = A[..., None, None] * delta * (1.0 - broadcast_mask(mask, delta))

        drift_matrix = B[..., None, None] * jnp.matmul(
            v_u[..., None], k_u[..., None].swapaxes(-1, -2)
        )
        write_matrix = beta[..., None, None] * jnp.matmul(
            v_s[..., None], k_s[..., None].swapaxes(-1, -2)
        )

        update_matrix = jnp.matmul(drift_matrix, delta) + write_matrix

        def binary_operator(lhs, rhs):
            a_i, b_i = lhs
            a_j, b_j = rhs
            return (jnp.matmul(a_i, a_j), jnp.matmul(b_i, a_j) + b_j)

        cummulative_decay, hidden_states = lax.associative_scan(
            jax.vmap(binary_operator), (decay_matrix, update_matrix), axis=1
        )
        hidden_states = (
            jnp.matmul(add_time_axis(carry), cummulative_decay) + hidden_states
        )

        x = (
            jnp.matmul(hidden_states, q_s[..., None])
            .squeeze(-1)
            .reshape(batch_size, sequence_length, hidden_dim)
        )

        x = nn.RMSNorm(dtype=self.dtype)(x)

        y = nn.Dense(
            features=in_features, kernel_init=self.kernel_init, bias_init=self.bias_init
        )(x)

        carry = hidden_states[:, -1]

        drift_norm = jnp.linalg.norm(drift_matrix, axis=(-2, -1))
        write_norm = jnp.linalg.norm(write_matrix, axis=(-2, -1))

        self.sow("intermediates", "drift_norm", drift_norm)
        self.sow("intermediates", "write_norm", write_norm)
        self.sow("intermediates", "gamma", B)
        self.sow("intermediates", "beta", beta)

        return carry, y


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


class KalmanNetBlock(nn.Module):
    head_dim: int
    num_heads: int
    num_actions: int
    ratio: int = 4
    kernel_init: Initializer = initializers.lecun_normal()
    bias_init: Initializer = initializers.zeros_init()
    param_dtype: Dtype = jnp.float32
    dtype: Optional[Dtype] = None

    @compact
    def __call__(
        self, inputs: Array, action: Array, mask: Array, carry: Array
    ) -> Tuple[Array, Array]:

        *_, in_features = inputs.shape

        skip = inputs

        x = nn.RMSNorm(dtype=self.dtype)(inputs)

        carry, x = KalmanNetLayer(
            head_dim=self.head_dim,
            num_heads=self.num_heads,
            num_actions=self.num_actions,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
            param_dtype=self.param_dtype,
            dtype=self.dtype,
        )(x, action, mask, carry)

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


class KalmanNet(SequenceModel):
    num_layers: int
    head_dim: int
    num_heads: int
    embedding_dim: int
    num_actions: int
    ratio: int = 4
    kernel_init: Initializer = initializers.lecun_normal()
    bias_init: Initializer = initializers.zeros_init()
    param_dtype: Dtype = jnp.float32
    dtype: Optional[Dtype] = None

    @compact
    def __call__(
        self,
        inputs: Array,
        action: Array,
        mask: Array,
        initial_carry: Optional[Tuple[Array, ...]] = None,
        **kwargs,
    ) -> Tuple[Tuple[Array, ...], Array]:

        new_carry = []

        if initial_carry is None:
            initial_carry = self.initialize_carry(None, inputs.shape)

        for i, carry_i in enumerate(initial_carry):
            carry_i, inputs = KalmanNetBlock(
                head_dim=self.head_dim,
                num_heads=self.num_heads,
                num_actions=self.num_actions,
                ratio=self.ratio,
                kernel_init=self.kernel_init,
                bias_init=self.bias_init,
                param_dtype=self.param_dtype,
                dtype=self.dtype,
                name=f"block_{i}",
            )(inputs, action, mask, carry_i)

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
