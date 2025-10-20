from functools import partial  # pylint: disable=g-importing-member
from typing import (
    Any,
    TypeVar,
    Callable,
)
import math

import jax
from jax import numpy as jnp
from jax import random

import flax.linen as nn
from flax.linen import initializers, LayerNorm, GroupNorm
from flax.linen.activation import sigmoid
from flax.linen.linear import Dense, default_kernel_init
from flax.linen.dtypes import promote_dtype
from flax.linen.module import compact, nowrap, Module
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


def f_bias_init(key, shape, dtype):
    """Initializes a weight matrix with a power law distribution."""
    num_heads, *_ = shape
    return jnp.linspace(3.0, 6.0, num_heads, dtype=dtype)


def small_init(embedding_dim: int):
    def init(key, shape, dtype):
        std = 1.0 / jnp.sqrt(embedding_dim)
        return jax.random.normal(key, shape, dtype) * std

    return init


def wang_init(embedding_dim: int, num_blocks: int):
    def init(key, shape, dtype):
        base = jnp.sqrt(2.0) / jnp.sqrt(embedding_dim)
        std = base / jnp.sqrt(max(1, num_blocks))
        return jax.random.normal(key, shape, dtype) * std

    return init


class BlockDiagonalDense(Module):

    block_size: int = 4
    use_bias: bool = True
    kernel_init: Initializer = default_kernel_init
    bias_init: Initializer = initializers.zeros_init()
    dtype: Dtype | None = None
    param_dtype: Dtype = jnp.float32

    @compact
    def __call__(self, x: Array) -> Array:
        *batch, features = x.shape
        num_heads = features // self.block_size
        head_dim = features // num_heads

        if head_dim % self.block_size != 0:
            raise ValueError(
                f"head_dim ({head_dim}) must be divisible by block_size ({self.block_size})."
            )
        block_dim = head_dim // self.block_size

        x = x.reshape(*batch, num_heads, block_dim, self.block_size)

        k = self.param(
            "kernel",
            self.kernel_init,
            (num_heads, block_dim, self.block_size, self.block_size),
            self.param_dtype,
        )
        b = (
            self.param(
                "bias",
                self.bias_init,
                (num_heads, block_dim, self.block_size),
                self.param_dtype,
            )
            if self.use_bias
            else None
        )

        x, k, b = promote_dtype(x, k, b, dtype=self.dtype)
        yh = jnp.einsum("...hbs,hbst->...hbt", x, k)
        if b is not None:
            yh = yh + b
        return yh.reshape(*batch, features)


class mLSTMCell(RNNCellBase):

    features: int
    num_heads: int = 4
    up_proj_factor: int = 2
    conv_kernel_size: int = 4
    forget_activation: str = "sigmoid"
    num_blocks: int = 1

    kernel_init: Initializer = default_kernel_init
    bias_init: Initializer = initializers.zeros_init()
    dropout_rate: float = 0.0
    dtype: Dtype | None = None
    param_dtype: Dtype = jnp.float32
    carry_init: Initializer = initializers.zeros_init()

    def _log_forget(self, f_raw: Array) -> Array:
        if self.forget_activation == "exp":
            # log(exp(f_raw)) = f_raw
            return f_raw
        elif self.forget_activation == "sigmoid":
            # log(sigmoid(x)) = -softplus(-x)
            return -jax.nn.softplus(-f_raw)
        else:
            raise ValueError("forget_activation must be 'exp' or 'sigmoid'.")

    @compact
    def __call__(self, carry, inputs):
        (C, n, m, conv_buf) = carry
        batch_dims = inputs.shape[:-1]
        up_features = self.features * self.up_proj_factor
        head_dim = up_features // self.num_heads
        assert up_features % self.num_heads == 0

        x_ln = LayerNorm(
            use_scale=True,
            use_bias=False,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="pre_ln",
        )(inputs)

        up = Dense(
            features=2 * up_features,
            use_bias=False,
            kernel_init=small_init(self.features),
            bias_init=self.bias_init,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="up_proj",
        )(x_ln)
        up_gate, up_core = jnp.split(up, 2, axis=-1)

        k_x = self.param(
            "conv_kernel",
            self.kernel_init,
            (up_features, self.conv_kernel_size),
            self.param_dtype,
        )

        causal_window = jnp.concatenate([up_core[..., None], conv_buf], axis=-1)
        conv_x = jnp.einsum("...fk,fk->...f", causal_window, k_x)
        conv_x = jax.nn.silu(conv_x)
        conv_buf_new = jnp.concatenate(
            [up_core[..., None], conv_buf[..., :-1]], axis=-1
        )

        linear_block_diagonal_dense = partial(
            BlockDiagonalDense,
            use_bias=False,
            kernel_init=small_init(self.features),
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )

        q = linear_block_diagonal_dense(name="q")(conv_x)
        k = linear_block_diagonal_dense(name="k")(conv_x)
        v = linear_block_diagonal_dense(name="v")(up_core)

        qkv = jnp.concatenate([q, k, v], axis=-1)

        def to_heads(x):
            return x.reshape(x.shape[:-1] + (self.num_heads, head_dim))

        q = to_heads(q)
        k = to_heads(k) / jnp.sqrt(jnp.array(head_dim, dtype=self.param_dtype))
        v = to_heads(v)

        gate_dense = partial(
            Dense,
            features=self.num_heads,
            use_bias=True,
            kernel_init=initializers.zeros_init(),
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )
        i_tilde = gate_dense(name="wi", bias_init=initializers.normal(stddev=0.1))(qkv)
        f_tilde = gate_dense(name="wf", bias_init=f_bias_init)(qkv)

        log_f = self._log_forget(f_tilde)

        m_new = jnp.maximum(log_f + m, i_tilde)
        i_prime = jnp.exp(i_tilde - m_new)
        f_prime = jnp.exp(log_f + m - m_new)

        C_new = f_prime[..., None, None] * C + i_prime[..., None, None] * (
            v[..., :, None] * k[..., None, :]
        )
        n_new = f_prime[..., None] * n + i_prime[..., None] * k

        denom = (
            jnp.maximum(jnp.abs(jnp.sum(n_new * q, axis=-1)), jnp.exp(-m_new)) + 1e-6
        )
        h_tilde = jnp.matmul(C_new, q[..., :, None])[..., :, 0] / denom[..., None]
        h_tilde = h_tilde.reshape(batch_dims + (up_features,))

        h_norm = GroupNorm(
            num_groups=self.num_heads,
            use_scale=True,
            use_bias=False,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="head_group_norm",
        )(h_tilde)

        lskip = self.param(
            "lskip",
            initializers.ones_init(),
            (up_features,),
            self.param_dtype,
        )
        h = h_norm + (lskip * conv_x)
        h = h * jax.nn.silu(up_gate)

        out = Dense(
            features=self.features,
            use_bias=False,
            kernel_init=wang_init(self.features, self.num_blocks),
            bias_init=self.bias_init,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="down_proj",
        )(h)

        out = nn.Dropout(
            rate=self.dropout_rate, deterministic=not self.has_rng("dropout")
        )(out)

        y = out + inputs
        return (C_new, n_new, m_new, conv_buf_new), y

    @staticmethod
    @nowrap
    def _initialize_carry(
        rng: PRNGKey,
        input_shape: tuple[int, ...],
        *,
        features,
        carry_init=initializers.zeros_init(),
        up_proj_factor=2,
        num_heads=4,
        conv_kernel_size=4,
        param_dtype=jnp.float32,
    ) -> tuple[Array, Array, Array, Array]:
        """To be called by xLSTMCell"""
        batch_dims = input_shape[:-1]
        up_features = features * up_proj_factor
        head_dim = up_features // num_heads
        key_c, key_n, key_m, key_conv_buf = random.split(rng, 4)
        C = carry_init(
            key_c, (batch_dims + (num_heads, head_dim, head_dim)), param_dtype
        )
        n = carry_init(key_n, (batch_dims + (num_heads, head_dim)), param_dtype)
        m = carry_init(key_m, (batch_dims + (num_heads,)), param_dtype)
        conv_buf = carry_init(
            key_conv_buf,
            batch_dims + (up_features, max(conv_kernel_size - 1, 0)),
            param_dtype,
        )
        return (C, n, m, conv_buf)

    @nowrap
    def initialize_carry(
        self, rng: PRNGKey, input_shape: tuple[int, ...]
    ) -> tuple[Array, Array, Array, Array]:
        batch_dims = input_shape[:-1]
        up_features = self.features * self.up_proj_factor
        head_dim = up_features // self.num_heads
        key_c, key_n, key_m, key_conv_buf = random.split(rng, 4)
        C = self.carry_init(
            key_c, (batch_dims + (self.num_heads, head_dim, head_dim)), self.param_dtype
        )
        n = self.carry_init(
            key_n, (batch_dims + (self.num_heads, head_dim)), self.param_dtype
        )
        m = self.carry_init(key_m, (batch_dims + (self.num_heads,)), self.param_dtype)
        conv_buf = self.carry_init(
            key_conv_buf,
            batch_dims + (up_features, max(self.conv_kernel_size - 1, 0)),
            self.param_dtype,
        )
        return (C, n, m, conv_buf)

    @property
    def num_feature_axes(self) -> int:
        return 1
