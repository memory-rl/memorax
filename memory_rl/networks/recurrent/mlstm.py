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


class mLSTMCell(RNNCellBase):
    r"""mLSTM cell from xLSTM (Beck et al., 2024)."""

    features: int
    num_heads: int = 1
    forget_activation: str = "exp"  # 'exp' or 'sigmoid'
    kernel_init: Initializer = default_kernel_init
    bias_init: Initializer = initializers.zeros_init()
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
        (C, n, h, m) = carry
        batch_dims = inputs.shape[:-1]

        head_dim = self.features // self.num_heads

        dense_vec = partial(
            Dense,
            use_bias=True,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )

        q = dense_vec(name="Wq", features=self.features)(inputs)
        k = dense_vec(name="Wk", features=self.features)(inputs)
        v = dense_vec(name="Wv", features=self.features)(inputs)
        o_tilde = dense_vec(name="Wo", features=self.features)(inputs)

        def to_heads(x):
            return x.reshape(x.shape[:-1] + (self.num_heads, head_dim))

        q = to_heads(q)
        k = to_heads(k) / jnp.sqrt(jnp.array(head_dim, dtype=self.param_dtype))
        v = to_heads(v)
        o = sigmoid(to_heads(o_tilde))

        gate_dense = partial(
            Dense,
            features=self.num_heads,
            use_bias=True,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )
        i_tilde = gate_dense(name="wi")(inputs)
        f_tilde = gate_dense(name="wf")(inputs)

        log_f = self._log_forget(f_tilde)

        m_new = jnp.maximum(log_f + m, i_tilde)
        i_prime = jnp.exp(i_tilde - m_new)
        f_prime = jnp.exp(log_f + m - m_new)

        C_new = f_prime[..., None, None] * C + i_prime[..., None, None] * (
            v[..., :, None] * k[..., None, :]
        )
        n_new = f_prime[..., None] * n + i_prime[..., None] * k

        denom = jnp.maximum(jnp.abs(jnp.sum(n_new * q, axis=-1)), jnp.asarray(1.0))
        h_tilde = jnp.matmul(C_new, q[..., :, None])[..., :, 0] / denom[..., None]

        h_new = o * h_tilde
        h_new = h_new.reshape(batch_dims + (self.features,))

        return (C_new, n_new, h_new, m_new), h_new

    @nowrap
    def initialize_carry(self, rng: PRNGKey, input_shape: tuple[int, ...]):
        batch_dims = input_shape[:-1]
        head_dim = self.features // self.num_heads

        key_C, key_n, key_h, key_m = random.split(rng, 4)
        C = self.carry_init(
            key_C, batch_dims + (self.num_heads, head_dim, head_dim), self.param_dtype
        )
        n = self.carry_init(
            key_n, batch_dims + (self.num_heads, head_dim), self.param_dtype
        )
        m = self.carry_init(key_m, batch_dims + (self.num_heads,), self.param_dtype)
        h = self.carry_init(key_h, batch_dims + (self.features,), self.param_dtype)
        return (C, n, h, m)

    @property
    def num_feature_axes(self) -> int:
        return 1


class BlockDiagonalDense(Module):

    num_heads: int
    block_size: int = 4
    use_bias: bool = True
    kernel_init: Initializer = default_kernel_init
    bias_init: Initializer = initializers.zeros_init()
    dtype: Dtype | None = None
    param_dtype: Dtype = jnp.float32

    @compact
    def __call__(self, x: Array) -> Array:
        *batch, features = x.shape
        if features % self.num_heads != 0:
            raise ValueError(
                f"features ({features}) must be divisible by num_heads ({self.num_heads})."
            )
        head_dim = features // self.num_heads

        if head_dim % self.block_size != 0:
            raise ValueError(
                f"head_dim ({head_dim}) must be divisible by block_size ({self.block_size})."
            )
        block_dim = head_dim // self.block_size

        x = x.reshape(*batch, self.num_heads, block_dim, self.block_size)

        k = self.param(
            "kernel",
            self.kernel_init,
            (self.num_heads, block_dim, self.block_size, self.block_size),
            self.param_dtype,
        )
        b = (
            self.param(
                "bias",
                self.bias_init,
                (self.num_heads, block_dim, self.block_size),
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


class mLSTMBlock(RNNCellBase):

    features: int
    num_heads: int = 4
    up_proj_factor: int = 2
    conv_kernel_size: int = 4
    forget_activation: str = "exp"

    kernel_init: Initializer = default_kernel_init
    bias_init: Initializer = initializers.zeros_init()
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
            use_bias=True,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="pre_ln",
        )(inputs)

        dense_up = partial(
            Dense,
            features=up_features,
            use_bias=True,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )
        up_gate = dense_up(name="up_gate")(x_ln)
        up_core = dense_up(name="up_core")(x_ln)

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
            num_heads=self.num_heads,
            use_bias=False,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )

        q = linear_block_diagonal_dense(name="q")(conv_x)
        k = linear_block_diagonal_dense(name="k")(conv_x)
        v = linear_block_diagonal_dense(name="v")(up_core)

        def to_heads(x):
            return x.reshape(x.shape[:-1] + (self.num_heads, head_dim))

        q = to_heads(q)
        k = to_heads(k) / jnp.sqrt(jnp.array(head_dim, dtype=self.param_dtype))
        v = to_heads(v)

        gate_dense = partial(
            Dense,
            features=self.num_heads,
            use_bias=True,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )
        i_tilde = gate_dense(name="wi")(conv_x)
        f_tilde = gate_dense(name="wf")(conv_x)

        log_f = self._log_forget(f_tilde)

        m_new = jnp.maximum(log_f + m, i_tilde)
        i_prime = jnp.exp(i_tilde - m_new)
        f_prime = jnp.exp(log_f + m - m_new)

        C_new = f_prime[..., None, None] * C + i_prime[..., None, None] * (
            v[..., :, None] * k[..., None, :]
        )
        n_new = f_prime[..., None] * n + i_prime[..., None] * k

        denom = jnp.maximum(jnp.abs(jnp.sum(n_new * q, axis=-1)), jnp.asarray(1.0))
        h_tilde = jnp.matmul(C_new, q[..., :, None])[..., :, 0] / denom[..., None]
        h_tilde = h_tilde.reshape(batch_dims + (up_features,))

        h_norm = GroupNorm(
            num_groups=self.num_heads,
            use_bias=True,
            use_scale=True,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="head_group_norm",
        )(h_tilde)

        lskip = Dense(
            features=up_features,
            use_bias=True,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="lskip",
        )(conv_x)
        h = h_norm + lskip
        h = jax.nn.swish(up_gate) * h

        out = Dense(
            features=self.features,
            use_bias=True,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="down_proj",
        )(h)

        y = out + inputs
        return (C_new, n_new, m_new, conv_buf_new), y

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
