from functools import partial
from typing import Any, TypeVar

import flax.linen as nn
import jax
from flax.linen import initializers
from flax.linen.module import compact, nowrap
from flax.linen.recurrent import RNNCellBase
from flax.typing import Array, Dtype, Initializer, PRNGKey
from jax import numpy as jnp
from jax import random

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


class sLSTMCell(RNNCellBase):
    """Scalar LSTM Cell compatible with Flax's RNNCellBase.

    This cell can be used directly with the RNN wrapper or nn.RNN.

    Attributes:
        features: Output feature dimension.
        hidden_dim: Hidden state dimension.
        num_heads: Number of attention heads.
        use_causal_conv: Whether to use causal convolution.
        conv_kernel_size: Kernel size for causal convolution.
        eps: Epsilon for numerical stability.
        dropout_rate: Dropout rate.
        dtype: Data type for computation.
        param_dtype: Data type for parameters.
    """

    features: int
    hidden_dim: int
    num_heads: int = 4
    use_causal_conv: bool = True
    conv_kernel_size: int = 4
    eps: float = 1e-6
    dropout_rate: float = 0.0
    dtype: Dtype | None = None
    param_dtype: Dtype = jnp.float32

    @compact
    def __call__(self, carry: tuple, inputs: Array) -> tuple[tuple, Array]:
        """Process a single timestep.

        Args:
            carry: Tuple of (cell_state, conv_state) where cell_state = (c, n, m, h)
            inputs: Input tensor of shape (B, F)

        Returns:
            Tuple of (new_carry, outputs) where outputs has shape (B, features)
        """
        cell_state, conv_state = carry
        c, n, m, h = cell_state

        B, *_ = inputs.shape
        head_dim = self.hidden_dim // self.num_heads
        if self.hidden_dim % self.num_heads != 0:
            raise ValueError(
                f"hidden_dim ({self.hidden_dim}) must be divisible by num_heads ({self.num_heads})."
            )

        # Project input to hidden dimension
        x_proj = nn.Dense(
            features=self.hidden_dim,
            use_bias=False,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="in_proj",
        )(inputs)

        # Causal convolution (operates on hidden_dim)
        if self.use_causal_conv:
            x = add_time_axis(x_proj)
            conv_state, conv_x = CausalConv1d(
                features=self.hidden_dim,
                kernel_size=self.conv_kernel_size,
                param_dtype=self.param_dtype,
            )(x, conv_state)
            conv_x_act = jax.nn.silu(conv_x)
            conv_x_act = remove_time_axis(conv_x_act)
        else:
            conv_x_act = x_proj

        # Input gate projections (from hidden_dim to hidden_dim)
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
        z = gate(name="z")(x_proj)
        o = gate(name="o")(x_proj)

        # Recurrent gate projections
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
            + recurrent_gate(name="ri")(h)
            + self.param(
                "i_bias",
                nn.initializers.zeros_init(),
                (self.hidden_dim,),
                self.param_dtype,
            )
        )
        f = (
            f
            + recurrent_gate(name="rf")(h)
            + self.param(
                "f_bias",
                powerlaw_init(self.num_heads, head_dim=head_dim),
                (self.hidden_dim,),
                self.param_dtype,
            )
        )
        z = (
            z
            + recurrent_gate(name="rz")(h)
            + self.param(
                "z_bias",
                nn.initializers.zeros_init(),
                (self.hidden_dim,),
                self.param_dtype,
            )
        )
        o = (
            o
            + recurrent_gate(name="ro")(h)
            + self.param(
                "o_bias",
                nn.initializers.zeros_init(),
                (self.hidden_dim,),
                self.param_dtype,
            )
        )

        # sLSTM recurrence
        o = jax.nn.sigmoid(o)
        log_f = -jax.nn.softplus(-f)
        m_new = jnp.where(jnp.all(n == 0.0), i, jnp.maximum(log_f + m, i))
        i_p = jnp.minimum(jnp.exp(i - m_new), jnp.ones_like(i))
        f_p = jnp.minimum(jnp.exp(log_f + m - m_new), jnp.ones_like(f))

        c_new = f_p * c + i_p * nn.tanh(z)
        n_new = f_p * n + i_p
        h_tilde = c_new / jnp.maximum(n_new, self.eps)
        h_new = o * h_tilde

        new_cell_state = (c_new, n_new, m_new, h_new)

        # Dropout
        y = nn.Dropout(
            rate=self.dropout_rate, deterministic=not self.has_rng("dropout")
        )(h_new)

        # Layer norm per head
        y = y.reshape(B, self.num_heads, 1, head_dim)
        y = MultiHeadLayerNorm(use_scale=True, use_bias=False)(y)

        # Output projection
        y = y.reshape(B, self.hidden_dim)
        y = nn.Dense(
            features=self.features,
            use_bias=False,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="out_proj",
        )(y)

        return (new_cell_state, conv_state), y

    @nowrap
    def initialize_carry(
        self,
        rng: PRNGKey,
        input_shape: tuple[int, ...],
    ) -> tuple:
        """Initialize the carry state.

        Args:
            rng: Random key for initialization.
            input_shape: Shape of input (batch_size, features).

        Returns:
            Initial carry tuple of (cell_state, conv_state).
        """
        *batch_dims, _ = input_shape
        carry_init = initializers.zeros_init()

        key_c, key_n, key_h, key_m, key_conv = random.split(rng, 5)
        mem_shape = (*batch_dims, self.hidden_dim)

        c = carry_init(key_c, mem_shape, self.param_dtype)
        n = carry_init(key_n, mem_shape, self.param_dtype)
        m = carry_init(key_m, mem_shape, self.param_dtype)
        h = carry_init(key_h, mem_shape, self.param_dtype)
        cell_state = (c, n, m, h)

        conv_state = carry_init(
            key_conv, (*batch_dims, self.conv_kernel_size, self.hidden_dim)
        )

        return cell_state, conv_state

    @property
    def num_feature_axes(self) -> int:
        return 1
