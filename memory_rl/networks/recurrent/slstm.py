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

from flax.linen import initializers, LayerNorm, GroupNorm
from flax.linen.activation import sigmoid, tanh
from flax.linen.dtypes import promote_dtype
from flax.linen.linear import Dense, default_kernel_init
from flax.linen.module import Module, compact, nowrap
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


class sLSTMCell(RNNCellBase):
    r"""sLSTM cell from xLSTM (Beck et al., 2024)."""

    features: int
    activation_fn: Callable[..., Any] = tanh
    forget_activation: str = "exp"  # 'exp' or 'sigmoid'
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

    @compact
    def __call__(
        self, carry: tuple[Array, Array, Array, Array], inputs: Array
    ) -> tuple[tuple[Array, Array, Array, Array], Array]:
        c, n, h, m = carry
        hidden_features = h.shape[-1]

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

        z_tilde = dense_i(name="iz")(inputs) + dense_h(name="hz")(h)
        i_tilde = dense_i(name="ii")(inputs) + dense_h(name="hi")(h)
        f_tilde = dense_i(name="if")(inputs) + dense_h(name="hf")(h)
        o_tilde = dense_i(name="io")(inputs) + dense_h(name="ho")(h)

        z = self.activation_fn(z_tilde)
        o = sigmoid(o_tilde)

        log_f = self._log_forget(f_tilde)
        m_new = jnp.maximum(log_f + m, i_tilde)
        i_prime = jnp.exp(i_tilde - m_new)
        f_prime = jnp.exp(log_f + m - m_new)

        c_new = f_prime * c + i_prime * z
        n_new = f_prime * n + i_prime

        h_tilde = c_new / jnp.maximum(n_new, self.eps)
        h_new = o * h_tilde

        return (c_new, n_new, h_new, m_new), h_new

    @nowrap
    def initialize_carry(
        self, rng: PRNGKey, input_shape: tuple[int, ...]
    ) -> tuple[Array, Array, Array, Array]:
        batch_dims = input_shape[:-1]
        key_c, key_n, key_h, key_m = random.split(rng, 4)
        mem_shape = batch_dims + (self.features,)
        c = self.carry_init(key_c, mem_shape, self.param_dtype)
        n = self.carry_init(key_n, mem_shape, self.param_dtype)
        h = self.carry_init(key_h, mem_shape, self.param_dtype)
        m = self.carry_init(key_m, mem_shape, self.param_dtype)
        return (c, n, h, m)

    @property
    def num_feature_axes(self) -> int:
        return 1


class BlockDiagonalDense(Module):

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
        xh = x.reshape(x.shape[:-1] + (self.num_heads, dh))

        k = self.param(
            "kernel", self.kernel_init, (self.num_heads, dh, dh), self.param_dtype
        )
        b = (
            self.param("bias", self.bias_init, (self.num_heads, dh), self.param_dtype)
            if self.use_bias
            else None
        )

        xh, k, b = promote_dtype(xh, k, b, dtype=self.dtype)
        yh = jnp.einsum("...hd,hdf->...hf", xh, k)
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
    r"""sLSTM Block (post up-projection)"""

    features: int
    num_heads: int = 4
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

    @compact
    def __call__(
        self,
        carry: tuple[Array, Array, Array, Array, Array],
        inputs: Array,
    ) -> tuple[tuple[Array, Array, Array, Array, Array], Array]:
        c, n, h, m, conv_buf = carry
        if self.features % self.num_heads != 0:
            raise ValueError(
                f"features ({self.features}) must be divisible by num_heads ({self.num_heads})."
            )

        x_ln = LayerNorm(
            use_scale=True,
            use_bias=True,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="pre_ln",
        )(inputs)

        if self.use_causal_conv:
            k_x = self.param(
                "conv_kernel",
                self.kernel_init,
                (self.conv_kernel_size, self.features),
                self.param_dtype,
            )
            causal_window = jnp.concatenate([x_ln[..., None, :], conv_buf], axis=-2)
            conv_x = jnp.sum(causal_window * k_x, axis=-2)
            conv_x = jax.nn.silu(conv_x)
            new_conv_buf = jnp.concatenate(
                [x_ln[..., None, :], conv_buf[..., :-1, :]], axis=-2
            )
        else:
            conv_x = 0.0
            new_conv_buf = conv_buf

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
            "z": linear_block_diagonal_dense(name="x_z"),
            "i": linear_block_diagonal_dense(name="x_i"),
            "f": linear_block_diagonal_dense(name="x_f"),
            "o": linear_block_diagonal_dense(name="x_o"),
        }
        lin_h = {
            "z": recurrent_block_diagonal_dense(name="h_z"),
            "i": recurrent_block_diagonal_dense(name="h_i"),
            "f": recurrent_block_diagonal_dense(name="h_f"),
            "o": recurrent_block_diagonal_dense(name="h_o"),
        }

        z_tilde = lin_x["z"](x_ln) + lin_h["z"](h)
        i_tilde = lin_x["i"](conv_x) + lin_h["i"](h)
        f_tilde = lin_x["f"](conv_x) + lin_h["f"](h)
        o_tilde = lin_x["o"](x_ln) + lin_h["o"](h)

        z = self.activation_fn(z_tilde)
        o = sigmoid(o_tilde)

        log_f = self._log_forget(f_tilde)
        m_new = jnp.maximum(log_f + m, i_tilde)
        i_p = jnp.exp(i_tilde - m_new)
        f_p = jnp.exp(log_f + m - m_new)

        c_new = f_p * c + i_p * z
        n_new = f_p * n + i_p
        h_tilde = c_new / jnp.maximum(n_new, self.eps)
        h_new = o * h_tilde

        h_norm = GroupNorm(
            num_groups=self.num_heads,
            use_bias=True,
            use_scale=True,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="head_group_norm",
        )(h_new)
        h = h_norm + inputs
        h_ln = LayerNorm(
            use_scale=True,
            use_bias=True,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="post_ln",
        )(h)

        out = GatedMLP(
            self.features,
            pf=self.mlp_pf,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="geglu",
        )(h_ln)

        y = out + h

        new_carry = (c_new, n_new, h_new, m_new, new_conv_buf)
        return new_carry, y

    @nowrap
    def initialize_carry(
        self, rng: PRNGKey, input_shape: tuple[int, ...]
    ) -> tuple[Array, Array, Array, Array, Array]:
        batch_dims = input_shape[:-1]
        key_c, key_n, key_h, key_m, key_conv_buf = random.split(rng, 5)
        mem_shape = batch_dims + (self.features,)
        c = self.carry_init(key_c, mem_shape, self.param_dtype)
        n = self.carry_init(key_n, mem_shape, self.param_dtype)
        h = self.carry_init(key_h, mem_shape, self.param_dtype)
        m = self.carry_init(key_m, mem_shape, self.param_dtype)
        buf_shape = batch_dims + (max(self.conv_kernel_size - 1, 0), self.features)
        convbuf = self.carry_init(key_conv_buf, buf_shape, self.param_dtype)
        return (c, n, h, m, convbuf)

    @property
    def num_feature_axes(self) -> int:
        return 1
