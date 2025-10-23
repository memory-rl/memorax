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
from flax.linen import initializers, LayerNorm
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

from memory_rl.networks.recurrent.utils import (
    BlockDiagonalDense,
    CausalConv1d,
    MultiHeadLayerNorm,
    wang_init,
    small_init,
    f_bias_init,
    add_time_axis,
)

A = TypeVar("A")
Carry = Any
CarryHistory = Any
Output = Any


class mLSTMCell(nn.Module):

    features: int
    num_heads: int
    up_proj_factor: int
    conv_kernel_size: int
    forget_activation: str
    num_blocks: int

    kernel_init: Initializer
    bias_init: Initializer
    dtype: Dtype | None
    param_dtype: Dtype
    carry_init: Initializer

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
    def __call__(self, q, k, v, carry):
        (c, n, m) = carry
        B, S, _ = q.shape
        head_dim = self.features // self.num_heads

        qkv = jnp.concatenate([q, k, v], axis=-1)

        def to_heads(x):
            x = x.reshape(B, self.num_heads, head_dim, -1)
            return x

        q = to_heads(q)
        k = to_heads(k) / jnp.sqrt(jnp.array(head_dim, dtype=self.param_dtype))
        v = to_heads(v)

        gate = partial(
            Dense,
            features=self.num_heads,
            kernel_init=initializers.zeros_init(),
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )
        i_tilde = gate(name="wi", bias_init=initializers.normal(stddev=0.1))(qkv)
        i_tilde = i_tilde.swapaxes(-1, -2)[..., None]

        f_tilde = gate(name="wf", bias_init=f_bias_init)(qkv)
        f_tilde = f_tilde.swapaxes(-1, -2)[..., None]

        log_f = self._log_forget(f_tilde)

        m_new = jnp.maximum(log_f + m, i_tilde)
        i_prime = jnp.exp(i_tilde - m_new)
        f_prime = jnp.exp(log_f + m - m_new)

        c_new = f_prime * c + i_prime * (k @ v.swapaxes(-1, -2))
        n_new = f_prime * n + i_prime * k

        nominator = q.swapaxes(-1, -2) @ c_new
        denominator = jnp.maximum(jnp.abs(q.swapaxes(-1, -2) @ n_new), jnp.exp(-m_new))
        h_tilde = nominator / (denominator + 1e-6)

        h_norm = MultiHeadLayerNorm(
            use_scale=True,
            use_bias=False,
        )(h_tilde)

        h_norm = h_norm.swapaxes(1, 2).reshape(B, S, -1)

        return (c_new, n_new, m_new), h_norm


class mLSTMLayer(nn.Module):

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
    param_dtype: jnp.dtype = jnp.float32
    carry_init: Initializer = initializers.zeros_init()

    @compact
    def __call__(self, carry, inputs):
        cell_state, conv_state = carry
        up_features = self.features * self.up_proj_factor
        assert up_features % self.num_heads == 0

        x = add_time_axis(inputs)

        up = Dense(
            features=2 * up_features,
            use_bias=False,
            kernel_init=small_init(self.features),
            bias_init=self.bias_init,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="up_proj",
        )(x)
        up_gate, up_core = jnp.split(up, 2, axis=-1)

        conv_state, conv_x = CausalConv1d(
            features=up_features,
            kernel_size=self.conv_kernel_size,
            param_dtype=self.param_dtype,
        )(up_core, conv_state)
        conv_x_act = jax.nn.silu(conv_x)

        linear_block_diagonal_dense = partial(
            BlockDiagonalDense,
            features=up_features,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )

        q = linear_block_diagonal_dense(name="q")(conv_x_act)
        k = linear_block_diagonal_dense(name="k")(conv_x_act)
        v = linear_block_diagonal_dense(name="v")(up_core)

        cell_state, h_tilde = mLSTMCell(
            features=up_features,
            num_heads=self.num_heads,
            up_proj_factor=self.up_proj_factor,
            conv_kernel_size=self.conv_kernel_size,
            forget_activation=self.forget_activation,
            num_blocks=self.num_blocks,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            carry_init=self.carry_init,
        )(q, k, v, cell_state)

        learnable_skip = self.param(
            "learnable_skip",
            initializers.ones_init(),
            (up_features,),
            self.param_dtype,
        )
        h = h_tilde + (learnable_skip * conv_x_act)
        h = h * jax.nn.silu(up_gate)

        y = Dense(
            features=self.features,
            use_bias=False,
            kernel_init=wang_init(self.features, self.num_blocks),
            bias_init=self.bias_init,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="down_proj",
        )(h)

        y = nn.Dropout(
            rate=self.dropout_rate, deterministic=not self.has_rng("dropout")
        )(y)

        return (cell_state, conv_state), y.squeeze(1)

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
    ) -> tuple:
        """To be called by xLSTMCell"""
        *batch_dims, _ = input_shape
        up_features = features * up_proj_factor
        head_dim = up_features // num_heads
        key_c, key_n, key_m, key_conv = random.split(rng, 4)
        C = carry_init(
            key_c, ((*batch_dims, num_heads, head_dim, head_dim)), param_dtype
        )
        n = carry_init(key_n, ((*batch_dims, num_heads, head_dim, 1)), param_dtype)
        m = carry_init(key_m, ((*batch_dims, num_heads, 1, 1)), param_dtype)
        cell_state = (C, n, m)

        conv_state = carry_init(key_conv, (*batch_dims, conv_kernel_size, up_features))

        return cell_state, conv_state
