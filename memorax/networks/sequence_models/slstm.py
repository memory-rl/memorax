from collections.abc import Callable
from functools import partial  # pylint: disable=g-importing-member
from typing import Any, TypeVar

import flax.linen as nn
import jax
from flax.linen import initializers
from flax.linen.activation import sigmoid, tanh
from flax.linen.linear import default_kernel_init
from flax.linen.module import compact, nowrap
from flax.typing import Array, Dtype, Initializer, PRNGKey
from jax import numpy as jnp
from jax import random

from memorax.networks.sequence_models.sequence_model import SequenceModel
from memorax.networks.sequence_models.utils import (BlockDiagonalDense,
                                                    CausalConv1d,
                                                    MultiHeadLayerNorm,
                                                    add_time_axis,
                                                    powerlaw_init,
                                                    remove_time_axis)

A = TypeVar("A")
Carry = Any
CarryHistory = Any
Output = Any


class sLSTMCell(nn.Module):
    hidden_dim: int
    num_heads: int
    eps: float
    dropout_rate: float
    dtype: Dtype | None
    param_dtype: Dtype

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

        recurrent_gate = partial(
            BlockDiagonalDense,
            self.hidden_dim,
            num_heads=self.num_heads,
            use_bias=False,
            kernel_init=nn.initializers.zeros_init(),
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )

        i = (
            i
            + recurrent_gate(name="i")(h)
            + self.param(
                "i_bias",
                nn.initializers.zeros_init(),
                (self.hidden_dim,),
                self.param_dtype,
            )
        )
        f = (
            f
            + recurrent_gate(name="f")(h)
            + self.param(
                "f_bias",
                powerlaw_init(
                    self.num_heads, head_dim=self.hidden_dim // self.num_heads
                ),
                (self.hidden_dim,),
                self.param_dtype,
            )
        )
        z = (
            z
            + recurrent_gate(name="z")(h)
            + self.param(
                "z_bias",
                nn.initializers.zeros_init(),
                (self.hidden_dim,),
                self.param_dtype,
            )
        )
        o = (
            o
            + recurrent_gate(name="o")(h)
            + self.param(
                "o_bias",
                nn.initializers.zeros_init(),
                (self.hidden_dim,),
                self.param_dtype,
            )
        )

        o = sigmoid(o)

        log_f = -jax.nn.softplus(-f)
        m_new = jnp.where(jnp.all(n == 0.0), i, jnp.maximum(log_f + m, i))
        i_p = jnp.minimum(jnp.exp(i - m_new), jnp.ones_like(i))
        f_p = jnp.minimum(jnp.exp(log_f + m - m_new), jnp.ones_like(f))

        c_new = f_p * c + i_p * nn.tanh(z)
        n_new = f_p * n + i_p
        h_tilde = c_new / jnp.maximum(n_new, self.eps)
        h_new = o * h_tilde

        return (c_new, n_new, m_new, h_new), h_new


class sLSTMLayer(nn.Module):
    features: int
    hidden_dim: int
    num_heads: int = 4
    use_causal_conv: bool = True
    conv_kernel_size: int = 4

    eps: float = 1e-6

    dropout_rate: float = 0.0
    dtype: Dtype | None = None
    param_dtype: jnp.dtype = jnp.float32

    @compact
    def __call__(
        self,
        carry: tuple,
        inputs: Array,
    ):
        cell_state, conv_state = carry

        B, *_ = inputs.shape
        head_dim = self.hidden_dim // self.num_heads
        if self.hidden_dim % self.num_heads != 0:
            raise ValueError(
                f"features ({self.hidden_dim}) must be divisible by num_heads ({self.num_heads})."
            )

        if self.use_causal_conv:
            x = add_time_axis(inputs)
            conv_state, conv_x = CausalConv1d(
                features=self.hidden_dim,
                kernel_size=self.conv_kernel_size,
                param_dtype=self.param_dtype,
            )(x, conv_state)
            conv_x_act = jax.nn.silu(conv_x)
            conv_x_act = remove_time_axis(conv_x_act)
        else:
            conv_x_act = inputs

        gate = partial(
            BlockDiagonalDense,
            self.hidden_dim,
            num_heads=self.num_heads,
            use_bias=False,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )

        i = gate(name="i")(conv_x_act)
        f = gate(name="f")(conv_x_act)
        z = gate(name="z")(inputs)
        o = gate(name="o")(inputs)

        cell_state, y = sLSTMCell(
            hidden_dim=self.hidden_dim,
            num_heads=self.num_heads,
            eps=self.eps,
            dropout_rate=self.dropout_rate,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )(i, f, z, o, cell_state)

        y = nn.Dropout(
            rate=self.dropout_rate, deterministic=not self.has_rng("dropout")
        )(y)

        y = y.reshape(B, self.num_heads, 1, head_dim)

        h_norm = MultiHeadLayerNorm(
            use_scale=True,
            use_bias=False,
        )(y)

        return (cell_state, conv_state), h_norm.reshape(B, self.features)

    @staticmethod
    @nowrap
    def _initialize_carry(
        rng: PRNGKey,
        input_shape: tuple[int, ...],
        *,
        hidden_dim,
        carry_init=initializers.zeros_init(),
        param_dtype=jnp.float32,
        conv_kernel_size=4,
    ) -> tuple:
        """To be called by xLSTMCell"""
        *batch_dims, _ = input_shape
        key_c, key_n, key_h, key_m, key_conv = random.split(rng, 5)
        mem_shape = (
            *batch_dims,
            hidden_dim,
        )
        c = carry_init(key_c, mem_shape, param_dtype)
        n = carry_init(key_n, mem_shape, param_dtype)
        m = carry_init(key_m, mem_shape, param_dtype)
        h = carry_init(key_h, mem_shape, param_dtype)
        cell_state = (c, n, m, h)

        conv_state = carry_init(key_conv, (*batch_dims, conv_kernel_size, hidden_dim))

        return cell_state, conv_state
