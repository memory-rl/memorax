from functools import partial  # pylint: disable=g-importing-member
from typing import (
    Any,
    TypeVar,
)
from collections.abc import Callable

import jax
from jax import numpy as jnp
from jax import random

import flax.linen as nn
from flax.linen import initializers, LayerNorm, GroupNorm
from flax.linen.activation import sigmoid, tanh
from flax.linen.dtypes import promote_dtype
from flax.linen.linear import Dense, default_kernel_init
from flax.linen.module import Module, compact, nowrap
from flax.linen.recurrent import RNNCellBase
import math
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


class sLSTMCell(RNNCellBase):
    r"""sLSTM cell from xLSTM (Beck et al., 2024).

    Vectorized forward pass (Appendix A.2):
        c_t = f_t ⊙ c_{t-1} + i_t ⊙ z_t
        n_t = f_t ⊙ n_{t-1} + i_t
        h_t = o_t ⊙ (c_t ⊙ n_t^{-1})

        z_t = tanh(W_z x_t + R_z h_{t-1} + b_z)
        i_raw = W_i x_t + R_i h_{t-1} + b_i
        f_raw = W_f x_t + R_f h_{t-1} + b_f
        o_t   = sigmoid(W_o x_t + R_o h_{t-1} + b_o)

    Exponential gating + stabilizer (Eqs. 12–17):
        i_t = exp(i_raw)  (stabilized as i'_t)
        f_t = exp(f_raw) or sigmoid(f_raw)  (choose via `forget_activation`)
        m_t = max( log(f_t) + m_{t-1}, i_raw )
        i'_t = exp(i_raw - m_t)
        f'_t = exp(log(f_t) + m_{t-1} - m_t)

    We use i'_t and f'_t in the recurrence (output & grads are invariant).

    Attributes:
      features: hidden size.
      activation_fn: φ for z_t (default: tanh).
      forget_activation: 'exp' or 'sigmoid' (default: 'exp').
      stabilize: whether to use the stabilizer state m_t (default: True).
      eps: epsilon to avoid division by zero in c_t / n_t.
      kernel_init, recurrent_kernel_init, bias_init, dtype, param_dtype, carry_init:
        follow the conventions in `LSTMCell`.
    """

    features: int
    activation_fn: Callable[..., Any] = tanh
    forget_activation: str = "exp"  # 'exp' or 'sigmoid'
    stabilize: bool = True
    eps: float = 1e-6

    kernel_init: Initializer = default_kernel_init
    recurrent_kernel_init: Initializer = initializers.orthogonal()
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

    def _apply_forget_no_stabilize(self, f_raw: Array) -> Array:
        if self.forget_activation == "exp":
            return jnp.exp(f_raw)
        else:
            return sigmoid(f_raw)

    @compact
    def __call__(
        self, carry: tuple[Array, Array, Array, Array], inputs: Array  # (c, n, h, m)
    ) -> tuple[tuple[Array, Array, Array, Array], Array]:
        c, n, h, m = carry
        hidden_features = h.shape[-1]

        # Input/recurrent affine transforms (match LSTMCell parameterization)
        dense_h = partial(
            Dense,
            features=hidden_features,
            use_bias=True,
            kernel_init=self.recurrent_kernel_init,
            bias_init=self.bias_init,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )
        dense_i = partial(
            Dense,
            features=hidden_features,
            use_bias=False,
            kernel_init=self.kernel_init,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )

        # Preactivations
        z_raw = dense_i(name="iz")(inputs) + dense_h(name="hz")(h)
        i_raw = dense_i(name="ii")(inputs) + dense_h(name="hi")(h)
        f_raw = dense_i(name="if")(inputs) + dense_h(name="hf")(h)
        o_raw = dense_i(name="io")(inputs) + dense_h(name="ho")(h)

        # Gates & candidate
        z = self.activation_fn(z_raw)
        o = sigmoid(o_raw)

        if self.stabilize:
            log_f = self._log_forget(f_raw)  # log f_t
            m_new = jnp.maximum(log_f + m, i_raw)  # Eq. (15)
            i_prime = jnp.exp(i_raw - m_new)  # Eq. (16)
            f_prime = jnp.exp(log_f + m - m_new)  # Eq. (17)
        else:
            m_new = m
            i_prime = jnp.exp(i_raw)
            f_prime = self._apply_forget_no_stabilize(f_raw)

        # Recurrences (Eqs. 34–36 with stabilized gates)
        new_c = f_prime * c + i_prime * z
        new_n = f_prime * n + i_prime

        # Hidden: identity on normalized state (Eq. 36); output gate applied
        h_tilde = new_c / jnp.maximum(new_n, self.eps)
        new_h = o * h_tilde

        return (new_c, new_n, new_h, m_new), new_h

    @nowrap
    def initialize_carry(
        self, rng: PRNGKey, input_shape: tuple[int, ...]
    ) -> tuple[Array, Array, Array, Array]:
        batch_dims = input_shape[:-1]
        key_c, key_n, key_h, key_m = random.split(rng, 4)
        mem_shape = batch_dims + (self.features,)
        c0 = self.carry_init(key_c, mem_shape, self.param_dtype)
        n0 = self.carry_init(key_n, mem_shape, self.param_dtype)
        h0 = self.carry_init(key_h, mem_shape, self.param_dtype)
        m0 = self.carry_init(key_m, mem_shape, self.param_dtype)
        return (c0, n0, h0, m0)

    @property
    def num_feature_axes(self) -> int:
        return 1


class BlockDiagonalDense(Module):
    """Block-diagonal (per-head) linear layer: split features into NH heads and
    apply a head-local Dense. Equivalent to a block-diagonal weight matrix."""

    num_heads: int
    use_bias: bool = True
    kernel_init: Initializer = default_kernel_init
    bias_init: Initializer = initializers.zeros_init()
    dtype: Dtype | None = None
    param_dtype: Dtype = jnp.float32

    @compact
    def __call__(self, x: Array) -> Array:
        features = x.shape[-1]
        if features % self.num_heads != 0:
            raise ValueError(
                f"features ({features}) must be divisible by num_heads ({self.num_heads})."
            )
        dh = features // self.num_heads
        xh = x.reshape(x.shape[:-1] + (self.num_heads, dh))  # (..., NH, dh)

        k = self.param(
            "kernel", self.kernel_init, (self.num_heads, dh, dh), self.param_dtype
        )
        b = None
        if self.use_bias:
            b = self.param(
                "bias", self.bias_init, (self.num_heads, dh), self.param_dtype
            )

        xh, k, b = promote_dtype(xh, k, b, dtype=self.dtype)
        yh = jnp.einsum("...hd,hdf->...hf", xh, k)  # (..., NH, dh)
        if b is not None:
            yh = yh + b
        return yh.reshape(x.shape[:-1] + (features,))


class GatedMLP(Module):
    """Gated MLP and projection factor (pf)."""

    features: int
    gate: Callable = jax.nn.gelu
    pf: float = 4.0 / 3.0
    kernel_init: Initializer = default_kernel_init
    bias_init: Initializer = initializers.zeros_init()
    dtype: Dtype | None = None
    param_dtype: Dtype = jnp.float32

    @compact
    def __call__(self, x: Array) -> Array:
        inner = int(math.ceil(self.pf * self.features))
        u = Dense(
            inner,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="up_a",
        )(x)
        v = Dense(
            inner,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="up_b",
        )(x)
        gated = self.gate(u) * v
        y = Dense(
            self.features,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="down",
        )(gated)
        return y


class sLSTMBlock(RNNCellBase):
    r"""sLSTM Block (post up-projection) as in Fig. 10 of xLSTM.

    Pipeline per time step t:
      x̃_t = LN(x_t)                                           (pre-LN residual)
      Add optional causal Conv4 (Swish) contributions to i/f gate inputs
      For i, f, o, z: block-diagonal (NH heads) input and recurrent linears
      Run stabilized sLSTM update on (c,n,h,m)
      h_t → GroupNorm (head-wise)
      y_t = x_t + GeGLU_updown(h_t)                            (residual output)

    Carry:
      (c_t, n_t, h_t, m_t, convbuf_t) with convbuf_t storing the last K-1 LN’d inputs.

    Args:
      features: model width (must be divisible by num_heads).
      num_heads: number of block-diagonal heads (NH).
      use_causal_conv: if True, add Swish+causal Conv4 inputs into i/f gates.
      conv_kernel_size: causal kernel size (default 4).
      activation_fn: φ for z (default tanh).
      forget_activation: 'exp' or 'sigmoid' for f gate (default 'exp').
      stabilize: use stabilizer state m_t (recommended True).
      eps: floor for n_t in h = c/n.
      kernel_init, recurrent_kernel_init, bias_init, dtype, param_dtype, carry_init: as in LSTMCell.
      mlp_pf: projection factor of the gated MLP (Fig. 10 uses 4/3).

    Output:
      The residual block output y_t (not the raw h_t).
    """

    features: int
    num_heads: int = 4
    num_layers: int = 1
    use_causal_conv: bool = True
    conv_kernel_size: int = 4

    activation_fn: Callable[..., Any] = tanh
    forget_activation: str = "exp"  # 'exp' or 'sigmoid'
    stabilize: bool = True
    eps: float = 1e-6
    mlp_pf: float = 4.0 / 3.0

    kernel_init: Initializer = default_kernel_init
    recurrent_kernel_init: Initializer = initializers.orthogonal()
    bias_init: Initializer = initializers.zeros_init()
    dtype: Dtype | None = None
    param_dtype: Dtype = jnp.float32
    carry_init: Initializer = initializers.zeros_init()

    def _log_forget(self, f_raw: Array) -> Array:
        if self.forget_activation == "exp":
            return f_raw
        elif self.forget_activation == "sigmoid":
            return -jax.nn.softplus(-f_raw)
        raise ValueError("forget_activation must be 'exp' or 'sigmoid'.")

    def _apply_forget_no_stabilize(self, f_raw: Array) -> Array:
        return jnp.exp(f_raw) if self.forget_activation == "exp" else sigmoid(f_raw)

    @compact
    def __call__(
        self,
        carry: tuple[Array, Array, Array, Array, Array],  # (c, n, h, m, convbuf)
        inputs: Array,
    ) -> tuple[tuple[Array, Array, Array, Array, Array], Array]:
        c, n, h, m, convbuf = carry  # Each: (batch, num_layers, ...)
        if self.features % self.num_heads != 0:
            raise ValueError(
                f"features ({self.features}) must be divisible by num_heads ({self.num_heads})."
            )
        x = inputs
        new_carry = []
        for layer in range(self.num_layers):
            c_l = c[:, layer]
            n_l = n[:, layer]
            h_l = h[:, layer]
            m_l = m[:, layer]
            convbuf_l = convbuf[:, layer]
            x_ln = LayerNorm(
                use_scale=True,
                use_bias=True,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                name=f"pre_ln_{layer}",
            )(x)
            if self.use_causal_conv:
                K = self.conv_kernel_size
                k_i = self.param(
                    f"conv_i_kernel_{layer}", self.kernel_init, (K, self.features), self.param_dtype
                )
                k_f = self.param(
                    f"conv_f_kernel_{layer}", self.kernel_init, (K, self.features), self.param_dtype
                )
                win = jnp.concatenate([x_ln[..., None, :], convbuf_l], axis=-2)
                conv_i = jnp.sum(win * k_i, axis=-2)
                conv_f = jnp.sum(win * k_f, axis=-2)
                conv_i = jax.nn.silu(conv_i)
                conv_f = jax.nn.silu(conv_f)
                new_convbuf_l = jnp.concatenate(
                    [x_ln[..., None, :], convbuf_l[..., :-1, :]], axis=-2
                )
            else:
                conv_i = conv_f = 0.0
                new_convbuf_l = convbuf_l
            linear_block_diagonal_dense = partial(
                BlockDiagonalDense,
                self.num_heads,
                use_bias=False,
                kernel_init=self.kernel_init,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
            )
            recurrent_block_diagonal_dense = partial(
                BlockDiagonalDense,
                self.num_heads,
                use_bias=True,
                kernel_init=self.recurrent_kernel_init,
                bias_init=self.bias_init,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
            )
            lin_x = {
                "z": linear_block_diagonal_dense(name=f"x_z_{layer}"),
                "i": linear_block_diagonal_dense(name=f"x_i_{layer}"),
                "f": linear_block_diagonal_dense(name=f"x_f_{layer}"),
                "o": linear_block_diagonal_dense(name=f"x_o_{layer}"),
            }
            lin_h = {
                "z": recurrent_block_diagonal_dense(name=f"h_z_{layer}"),
                "i": recurrent_block_diagonal_dense(name=f"h_i_{layer}"),
                "f": recurrent_block_diagonal_dense(name=f"h_f_{layer}"),
                "o": recurrent_block_diagonal_dense(name=f"h_o_{layer}"),
            }
            z_raw = lin_x["z"](x_ln) + lin_h["z"](h_l)
            i_raw = lin_x["i"](x_ln) + lin_h["i"](h_l) + conv_i
            f_raw = lin_x["f"](x_ln) + lin_h["f"](h_l) + conv_f
            o_raw = lin_x["o"](x_ln) + lin_h["o"](h_l)
            z = self.activation_fn(z_raw)
            o = sigmoid(o_raw)
            if self.stabilize:
                log_f = self._log_forget(f_raw)
                m_new_l = jnp.maximum(log_f + m_l, i_raw)
                i_p = jnp.exp(i_raw - m_new_l)
                f_p = jnp.exp(log_f + m_l - m_new_l)
            else:
                m_new_l = m_l
                i_p = jnp.exp(i_raw)
                f_p = self._apply_forget_no_stabilize(f_raw)
            c_new_l = f_p * c_l + i_p * z
            n_new_l = f_p * n_l + i_p
            h_tilde = c_new_l / jnp.maximum(n_new_l, self.eps)
            h_new_l = o * h_tilde
            h_norm = GroupNorm(
                num_groups=self.num_heads,
                use_bias=True,
                use_scale=True,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                name=f"head_group_norm_{layer}",
            )(h_new_l)
            h_res = h_norm + x
            h_ln = LayerNorm(
                use_scale=True,
                use_bias=True,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                name=f"post_ln_{layer}",
            )(h_res)
            mlp_out = GatedMLP(
                self.features,
                pf=self.mlp_pf,
                kernel_init=self.kernel_init,
                bias_init=self.bias_init,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                name=f"geglu_{layer}",
            )(h_ln)
            y = mlp_out + h_res
            # Prepare carry for this layer
            new_carry.append((c_new_l, n_new_l, h_new_l, m_new_l, new_convbuf_l))
            # Output for next layer
            x = y
        # Stack carry arrays: each (batch, features) -> (batch, num_layers, features)
        c_new = jnp.stack([nc[0] for nc in new_carry], axis=1)
        n_new = jnp.stack([nc[1] for nc in new_carry], axis=1)
        h_new = jnp.stack([nc[2] for nc in new_carry], axis=1)
        m_new = jnp.stack([nc[3] for nc in new_carry], axis=1)
        convbuf_new = jnp.stack([nc[4] for nc in new_carry], axis=1)
        return (c_new, n_new, h_new, m_new, convbuf_new), x
    
    @nowrap
    def initialize_carry(
        self, rng: PRNGKey, input_shape: tuple[int, ...]
    ) -> tuple[Array, Array, Array, Array, Array]:
        batch_dims = input_shape[:-1]
        k1, k2, k3, k4, k5 = random.split(rng, 5)
        mem_shape = batch_dims + (self.num_layers, self.features)
        c0 = self.carry_init(k1, mem_shape, self.param_dtype)
        n0 = self.carry_init(k2, mem_shape, self.param_dtype)
        h0 = self.carry_init(k3, mem_shape, self.param_dtype)
        m0 = self.carry_init(k4, mem_shape, self.param_dtype)
        K = self.conv_kernel_size
        buf_shape = batch_dims + (self.num_layers, max(K - 1, 0), self.features)
        convbuf0 = self.carry_init(k5, buf_shape, self.param_dtype)
        return (c0, n0, h0, m0, convbuf0)

    @property
    def num_feature_axes(self) -> int:
        return 1


if __name__ == "__main__":
    batch_size = 2
    seq_length = 10
    embedding_dim = 64

    rnn = nn.RNN(
        sLSTMBlock(
            features=embedding_dim,
            num_heads=4,
            num_layers=2,  # Example: 2 layers
        )
    )

    x = jnp.ones((batch_size, seq_length, embedding_dim))
    variables = rnn.init(jax.random.PRNGKey(0), x)
    y = rnn.apply(variables, x)
    assert y.shape == (
        batch_size,
        seq_length,
        embedding_dim,
    ), f"Unexpected output shape: {y.shape}"
