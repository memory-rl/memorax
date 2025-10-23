from functools import partial  # pylint: disable=g-importing-member
import math
from typing import (
    Any,
    TypeVar,
)
from collections.abc import Callable

import jax
from jax import numpy as jnp
from jax import random

import flax.linen as nn
from flax.linen import initializers, GroupNorm
from flax.linen.activation import sigmoid, tanh
from flax.linen.linear import Dense, default_kernel_init
from flax.linen.module import Module, compact, nowrap
from flax.typing import (
    Array,
    PRNGKey,
    Dtype,
    Initializer,
)

from memory_rl.networks.recurrent.utils import CausalConv1d, MultiHeadLayerNorm, add_time_axis, f_bias_init, BlockDiagonalDense, remove_time_axis

A = TypeVar("A")
Carry = Any
CarryHistory = Any
Output = Any

class sLSTMCell(nn.Module):
    features: int
    num_heads: int
    use_causal_conv: bool
    conv_kernel_size: int

    activation_fn: Callable[..., Any]
    forget_activation: str
    eps: float
    mlp_pf: float

    kernel_init: Initializer
    recurrent_kernel_init: Initializer
    bias_init: Initializer
    dropout_rate: float
    dtype: Dtype | None
    param_dtype: Dtype
    carry_init: Initializer

    def _log_forget(self, f_raw: Array) -> Array:
        if self.forget_activation == "exp":
            return f_raw
        elif self.forget_activation == "sigmoid":
            return -jax.nn.softplus(-f_raw)
        raise ValueError("forget_activation must be 'exp' or 'sigmoid'.")

    @compact
    def __call__(
        self,
        i,
        f,
        z,
        o,
        carry: tuple,
    ):
        c, n, m, h = carry

        recurrent_block_diagonal_dense = partial(
            BlockDiagonalDense,
            self.features,
            use_bias=False,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )

        i = i + recurrent_block_diagonal_dense(name="i")(h) + self.param("i_bias", self.bias_init, (self.features,), self.param_dtype)
        f = f + recurrent_block_diagonal_dense(name="f")(h) + self.param("f_bias", f_bias_init, (self.features,), self.param_dtype)
        z = z + recurrent_block_diagonal_dense(name="z")(h) + self.param("z_bias", self.bias_init, (self.features,), self.param_dtype)
        o = o + recurrent_block_diagonal_dense(name="o")(h) + self.param("o_bias", self.bias_init, (self.features,), self.param_dtype)

        o = sigmoid(o)

        log_f = self._log_forget(f)
        m_new = jnp.where(jnp.all(n == 0.0), i, jnp.maximum(log_f + m, i))
        i_p = jnp.minimum(jnp.exp(i - m_new), jnp.ones_like(i))
        f_p = jnp.minimum(jnp.exp(log_f + m - m_new), jnp.ones_like(f))

        c_new = f_p * c + i_p * self.activation_fn(z)
        n_new = f_p * n + i_p
        h_tilde = c_new / jnp.maximum(n_new, self.eps)
        h_new = o * h_tilde

        return (c_new, n_new, m_new, h_new), h_new


class sLSTMLayer(nn.Module):
    features: int
    num_heads: int = 4
    use_causal_conv: bool = True
    conv_kernel_size: int = 4

    activation_fn: Callable[..., Any] = tanh
    forget_activation: str = "sigmoid"  # 'exp' or 'sigmoid'
    eps: float = 1e-6
    mlp_pf: float = 4.0 / 3.0

    kernel_init: Initializer = default_kernel_init
    recurrent_kernel_init: Initializer = initializers.zeros_init()
    bias_init: Initializer = initializers.zeros_init()
    dropout_rate: float = 0.0
    dtype: Dtype | None = None
    param_dtype: jnp.dtype = jnp.float32
    carry_init: Initializer = initializers.zeros_init()

    def _log_forget(self, f_raw: Array) -> Array:
        if self.forget_activation == "exp":
            return f_raw
        elif self.forget_activation == "sigmoid":
            return -jax.nn.softplus(-f_raw)
        raise ValueError("forget_activation must be 'exp' or 'sigmoid'.")

    @compact
    def __call__(
        self,
        carry: tuple,
        inputs: Array,
    ):
        cell_state, conv_state = carry

        B, *_ = inputs.shape
        head_dim = self.features // self.num_heads
        if self.features % self.num_heads != 0:
            raise ValueError(
                f"features ({self.features}) must be divisible by num_heads ({self.num_heads})."
            )


        if self.use_causal_conv:
            x = add_time_axis(inputs)
            conv_state, conv_x = CausalConv1d(features=self.features, kernel_size=self.conv_kernel_size, param_dtype=self.param_dtype)(x, conv_state)
            conv_x_act = jax.nn.silu(conv_x)
            conv_x_act = remove_time_axis(conv_x_act)
        else:
            conv_x_act = inputs


        linear_block_diagonal_dense = partial(
            BlockDiagonalDense,
            self.features,
            use_bias=False,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )

        i = linear_block_diagonal_dense(name="i")(conv_x_act)
        f = linear_block_diagonal_dense(name="f")(conv_x_act)
        z = linear_block_diagonal_dense(name="z")(inputs)
        o = linear_block_diagonal_dense(name="o")(inputs)

        cell_state, y = sLSTMCell(
            features=self.features,
            num_heads=self.num_heads,
            use_causal_conv=self.use_causal_conv,
            conv_kernel_size=self.conv_kernel_size,
            activation_fn=self.activation_fn,
            forget_activation=self.forget_activation,
            eps=self.eps,
            mlp_pf=self.mlp_pf,
            kernel_init=self.kernel_init,
            recurrent_kernel_init=self.recurrent_kernel_init,
            bias_init=self.bias_init,
            dropout_rate=self.dropout_rate,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            carry_init=self.carry_init,
        )(i, f, z, o, cell_state)

        y = y.reshape(B, self.num_heads, 1, head_dim)

        h_dropout = nn.Dropout(
            rate=self.dropout_rate, deterministic=not self.has_rng("dropout")
        )(y)

        h_norm = MultiHeadLayerNorm(
            use_scale=True,
            use_bias=False,
        )(h_dropout)

        return (cell_state, conv_state), h_norm.reshape(B, self.features)

    @staticmethod
    @nowrap
    def _initialize_carry(
        rng: PRNGKey,
        input_shape: tuple[int, ...],
        *,
        features,
        carry_init=initializers.zeros_init(),
        param_dtype=jnp.float32,
        conv_kernel_size=4,
    ) -> tuple:
        """To be called by xLSTMCell"""
        batch_dims = input_shape[:-1]
        key_c, key_n, key_h, key_m, key_conv = random.split(rng, 5)
        mem_shape = batch_dims + (features,)
        c = carry_init(key_c, mem_shape, param_dtype)
        n = carry_init(key_n, mem_shape, param_dtype)
        m = carry_init(key_m, mem_shape, param_dtype)
        h = carry_init(key_h, mem_shape, param_dtype)
        cell_state = (c, n, m, h)

        conv_state = carry_init(key_conv, (*batch_dims, conv_kernel_size, features))
        return cell_state, conv_state

