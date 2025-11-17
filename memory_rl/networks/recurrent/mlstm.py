from functools import partial  # pylint: disable=g-importing-member
from typing import (
    Any,
    TypeVar,
)

import jax
from jax import numpy as jnp
from jax import random

import flax.linen as nn
from flax.linen import initializers
from flax.linen.linear import Dense, default_kernel_init
from flax.linen.module import compact, nowrap
from flax.typing import (
    PRNGKey,
    Dtype,
    Initializer,
)

from memory_rl.networks.recurrent.utils import (
    BlockDiagonalDense,
    CausalConv1d,
    MultiHeadLayerNorm,
    kaiming_uniform,
    linspace_init,
    remove_time_axis,
    wang_init,
    small_init,
    add_time_axis,
)

A = TypeVar("A")
Carry = Any
CarryHistory = Any
Output = Any


class mLSTMCell(nn.Module):

    features: int
    num_heads: int

    kernel_init: Initializer
    bias_init: Initializer
    dtype: Dtype | None
    param_dtype: Dtype
    carry_init: Initializer

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
            name="gate"
        )
        i_tilde = gate(name="wi", bias_init=initializers.normal(stddev=0.1))(qkv)
        i_tilde = i_tilde.swapaxes(-1, -2)[..., None]
        f_tilde = gate(name="wf", bias_init=linspace_init(start=3.0, stop=6.0))(qkv)
        f_tilde = f_tilde.swapaxes(-1, -2)[..., None]

        log_f = -jax.nn.softplus(-f_tilde)

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
    up_proj_factor: float = 2.0
    num_heads: int = 4
    conv_kernel_size: int = 4
    num_blocks: int = 1
    block_size: int = 4

    kernel_init: Initializer = default_kernel_init
    bias_init: Initializer = initializers.zeros_init()
    dropout_rate: float = 0.0
    dtype: Dtype | None = None
    param_dtype: jnp.dtype = jnp.float32
    carry_init: Initializer = initializers.zeros_init()

    @compact
    def __call__(self, carry, inputs):
        cell_state, conv_state = carry
        *_, in_features = inputs.shape
        up_features: int = in_features * self.up_proj_factor

        x = add_time_axis(inputs)

        up_proj = Dense(
            features=2 * up_features,
            use_bias=False,
            kernel_init=small_init(in_features),
            bias_init=self.bias_init,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="up_proj",
        )(x)
        x, z = jnp.split(up_proj, 2, axis=-1)

        conv_state, conv_x = CausalConv1d(
            features=up_features,
            kernel_size=self.conv_kernel_size,
            param_dtype=self.param_dtype,
        )(x, conv_state)
        conv_x_act = jax.nn.silu(conv_x)

        proj = partial(
            BlockDiagonalDense,
            features=up_features,
            num_heads=self.num_heads,
            use_bias=False,
            kernel_init=small_init(self.features),
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )

        q = proj(name="q_proj")(conv_x_act)
        k = proj(name="k_proj")(conv_x_act)
        v = proj(name="v_proj")(x)

        cell_state, h_tilde = mLSTMCell(
            features=up_features,
            num_heads=self.num_heads,
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
        h = h * jax.nn.silu(z)

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

        y = remove_time_axis(y)

        return (cell_state, conv_state), y

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
