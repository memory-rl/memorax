from functools import partial  # pylint: disable=g-importing-member
from typing import Any, TypeVar

import flax.linen as nn
import jax
from flax.linen import initializers
from flax.linen.linear import Dense, default_kernel_init
from flax.linen.module import compact, nowrap
from flax.typing import Dtype, Initializer, PRNGKey
from jax import numpy as jnp
from jax import random

from memorax.networks.sequence_models.sequence_model import SequenceModel
from memorax.networks.sequence_models.utils import (
    BlockDiagonalDense,
    MultiHeadLayerNorm,
    ParallelCausalConv1d,
    linspace_init,
    small_init,
    wang_init,
)

A = TypeVar("A")
Carry = Any
CarryHistory = Any
Output = Any


class mLSTMCell(nn.Module):

    hidden_dim: int
    num_heads: int

    kernel_init: Initializer
    bias_init: Initializer
    dtype: Dtype | None
    param_dtype: Dtype
    carry_init: Initializer

    @compact
    def __call__(self, q, k, v, mask, state):
        B, S, _ = q.shape
        NH = self.num_heads
        DH = self.hidden_dim // NH

        prev_c, prev_n, prev_m = state

        qkv = jnp.concatenate([q, k, v], axis=-1)

        gate = partial(
            Dense,
            features=self.num_heads,
            kernel_init=initializers.zeros_init(),
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )

        i_gate = gate(name="wi", bias_init=initializers.normal(stddev=0.1))(qkv)
        i_gate = i_gate.reshape(B, NH, S)
        f_gate = gate(name="wf", bias_init=linspace_init(start=3.0, stop=6.0))(qkv)
        f_gate = f_gate.reshape(B, NH, S)

        def to_heads(x):
            x = x.reshape(B, NH, S, DH)
            return x

        q = to_heads(q) / jnp.sqrt(DH)
        k = to_heads(k)
        v = to_heads(v)

        log_f_gate = -jax.nn.softplus(-f_gate)
        log_f_gate_cumsum = jnp.cumsum(log_f_gate, axis=-1)

        log_f_gate_matrix = jnp.expand_dims(log_f_gate_cumsum, axis=-1) - jnp.swapaxes(
            jnp.expand_dims(log_f_gate_cumsum, axis=-1), -1, -2
        )

        causal_mask = nn.make_causal_mask(jnp.ones((B, S)))
        episode_indices = jnp.cumsum(mask, axis=-1)
        attention_mask = nn.make_attention_mask(
            episode_indices, episode_indices, pairwise_fn=jnp.equal
        )
        mask = nn.combine_masks(attention_mask, causal_mask)
        log_f_gate_matrix = jnp.where(
            mask,
            -jnp.inf,
            log_f_gate_matrix,
        )

        log_d_matrix = log_f_gate_matrix + jnp.expand_dims(i_gate, axis=-1)
        max_d = jnp.max(log_d_matrix, axis=-1, keepdims=True)

        state_decay_curve = jnp.expand_dims(log_f_gate_cumsum, axis=-1) + prev_m
        stabilization = jnp.maximum(max_d, state_decay_curve)
        decay = jnp.exp(state_decay_curve - stabilization)

        inter_c = (q * decay) @ prev_c
        inter_n = (q * decay) @ prev_n

        d_matrix = jnp.exp(log_d_matrix - stabilization)

        qk_matrix = q @ jnp.swapaxes(k, -1, -2)
        e_matrix = qk_matrix * d_matrix

        normalizer = jnp.maximum(
            jnp.abs(jnp.sum(e_matrix, axis=-1, keepdims=True)) + inter_n,
            jnp.exp(-stabilization),
        )
        intra = (e_matrix / (normalizer + 1e-6)) @ v
        inter = inter_c / (normalizer + 1e-6)
        output = (intra + inter).reshape(B, NH, S, DH)

        final_decay = log_f_gate_cumsum[..., -1, None]

        log_gates = i_gate - log_f_gate_cumsum
        prev_m = prev_m.reshape(B, NH, 1)
        m = jnp.maximum(
            final_decay + prev_m,
            jnp.max(log_gates + final_decay, axis=-2, keepdims=True),
        )

        prev_decay = jnp.exp(prev_m + final_decay - m)
        kv_decay = jnp.exp(log_gates + final_decay - m)
        c = (
            prev_c * jnp.expand_dims(prev_decay, axis=-1)
            + jnp.swapaxes(k * jnp.expand_dims(kv_decay, -1), -1, -2) @ v
        )
        n = prev_n * prev_decay + jnp.sum(k * kv_decay[..., None, None], axis=-1)

        return (c, n, m), output


class mLSTMLayer(nn.Module):

    features: int
    hidden_dim: int
    up_proj_factor: float = 2
    block_size: int = 4
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
    def __call__(self, inputs, mask, carry):
        _, S, *_ = inputs.shape
        conv_state, cell_state = carry

        hidden_dim: int = self.hidden_dim * self.up_proj_factor

        up_proj = Dense(
            features=2 * hidden_dim,
            use_bias=False,
            kernel_init=small_init(self.hidden_dim),
            bias_init=self.bias_init,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="up_proj",
        )(inputs)
        x, z = jnp.split(up_proj, 2, axis=-1)

        x = jnp.concatenate([conv_state, x], axis=1)
        conv_x = ParallelCausalConv1d(
            features=hidden_dim,
        )(x)
        conv_x_act = jax.nn.silu(conv_x)[:, -S:, :]
        conv_state = x[:, -self.conv_kernel_size :, :]
        x = x[:, -S:, :]

        projection = partial(
            BlockDiagonalDense,
            features=hidden_dim,
            num_heads=hidden_dim // self.block_size,
            use_bias=False,
            kernel_init=small_init(self.features),
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )

        q = projection(name="q")(conv_x_act)
        k = projection(name="k")(conv_x_act)
        v = projection(name="v")(x)

        cell_state, h_tilde = mLSTMCell(
            hidden_dim=hidden_dim,
            num_heads=self.num_heads,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            carry_init=self.carry_init,
        )(q, k, v, mask, cell_state)

        learnable_skip = self.param(
            "learnable_skip",
            initializers.ones_init(),
            (hidden_dim,),
            self.param_dtype,
        )
        h = h_tilde + (learnable_skip * conv_x_act)
        h = h * jax.nn.silu(z)

        y = Dense(
            features=self.features,
            use_bias=False,
            kernel_init=wang_init(self.hidden_dim, self.num_blocks),
            bias_init=self.bias_init,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="down_proj",
        )(h)

        y = nn.Dropout(
            rate=self.dropout_rate, deterministic=not self.has_rng("dropout")
        )(y)

        return (conv_state, cell_state), y[:, -S:, :]

    @staticmethod
    @nowrap
    def _initialize_carry(
        rng: PRNGKey,
        input_shape: tuple[int, ...],
        *,
        hidden_dim,
        carry_init=initializers.zeros_init(),
        up_proj_factor=2,
        num_heads=4,
        conv_kernel_size=4,
        param_dtype=jnp.float32,
    ) -> tuple:
        """To be called by xLSTMCell"""
        *batch_dims, _ = input_shape
        hidden_dim = hidden_dim * up_proj_factor
        head_dim = hidden_dim // num_heads
        key_c, key_n, key_m, key_conv = random.split(rng, 4)

        conv_state = carry_init(key_conv, (*batch_dims, conv_kernel_size, hidden_dim))

        C = carry_init(
            key_c, ((*batch_dims, num_heads, head_dim, head_dim)), param_dtype
        )
        n = carry_init(key_n, ((*batch_dims, num_heads, head_dim, 1)), param_dtype)
        m = carry_init(key_m, ((*batch_dims, num_heads, 1, 1)), param_dtype)
        cell_state = (C, n, m)

        return conv_state, cell_state
