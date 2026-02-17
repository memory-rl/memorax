from functools import partial
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.linen import initializers
from flax.linen.recurrent import RNNCellBase
from flax.typing import Dtype
from jax import random

from memorax.utils.typing import Array

from .utils import (
    BlockDiagonalDense,
    CausalConv1d,
    MultiHeadLayerNorm,
    add_time_axis,
    powerlaw_init,
    remove_time_axis,
)


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
    dtype: Optional[Dtype] = None
    param_dtype: Dtype = jnp.float32

    def setup(self):
        head_dim = self.hidden_dim // self.num_heads
        if self.hidden_dim % self.num_heads != 0:
            raise ValueError(
                f"hidden_dim ({self.hidden_dim}) must be divisible by num_heads ({self.num_heads})."
            )

        self.input_projection = nn.Dense(
            features=self.hidden_dim,
            use_bias=False,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )

        if self.use_causal_conv:
            self.causal_conv = CausalConv1d(
                features=self.hidden_dim,
                kernel_size=self.conv_kernel_size,
                param_dtype=self.param_dtype,
            )

        gate = partial(
            BlockDiagonalDense,
            self.hidden_dim,
            num_heads=self.num_heads,
            use_bias=False,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )

        self.i_gate = gate()
        self.f_gate = gate()
        self.z_gate = gate()
        self.o_gate = gate()

        recurrent_gate = partial(
            BlockDiagonalDense,
            self.hidden_dim,
            num_heads=self.num_heads,
            use_bias=False,
            kernel_init=nn.initializers.zeros_init(),
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )

        self.ri = recurrent_gate()
        self.rf = recurrent_gate()
        self.rz = recurrent_gate()
        self.ro = recurrent_gate()

        self.i_bias = self.param(
            "i_bias",
            nn.initializers.zeros_init(),
            (self.hidden_dim,),
            self.param_dtype,
        )
        self.f_bias = self.param(
            "f_bias",
            powerlaw_init(self.num_heads, head_dim=head_dim),
            (self.hidden_dim,),
            self.param_dtype,
        )
        self.z_bias = self.param(
            "z_bias",
            nn.initializers.zeros_init(),
            (self.hidden_dim,),
            self.param_dtype,
        )
        self.o_bias = self.param(
            "o_bias",
            nn.initializers.zeros_init(),
            (self.hidden_dim,),
            self.param_dtype,
        )

        self.drop = nn.Dropout(rate=self.dropout_rate)
        self.norm = MultiHeadLayerNorm(use_scale=True, use_bias=False)
        self.output_projection = nn.Dense(
            features=self.features,
            use_bias=False,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )

    def __call__(self, carry: Tuple, inputs: Array) -> Tuple[Tuple, Array]:
        """Process a single timestep.

        Args:
            carry: Tuple of ((c, n, m, h), buffer)
            inputs: Input tensor of shape (B, F)

        Returns:
            Tuple of (new_carry, outputs) where outputs has shape (B, features)
        """
        (c, n, m, h), buffer = carry

        B, *_ = inputs.shape
        head_dim = self.hidden_dim // self.num_heads

        x = self.input_projection(inputs)

        if self.use_causal_conv:
            buffer, u = self.causal_conv(add_time_axis(x), buffer)
            u = jax.nn.silu(remove_time_axis(u))
        else:
            u = x

        i = self.i_gate(u) + self.ri(h) + self.i_bias
        f = self.f_gate(u) + self.rf(h) + self.f_bias
        z = self.z_gate(x) + self.rz(h) + self.z_bias
        o = jax.nn.sigmoid(self.o_gate(x) + self.ro(h) + self.o_bias)

        log_f = -jax.nn.softplus(-f)
        log_f_plus_m = log_f + m
        m = jnp.where(jnp.all(n == 0.0), i, jnp.maximum(log_f_plus_m, i))
        i = jnp.minimum(jnp.exp(i - m), jnp.ones_like(i))
        f = jnp.minimum(jnp.exp(log_f_plus_m - m), jnp.ones_like(f))

        c = f * c + i * nn.tanh(z)
        n = f * n + i
        h = o * (c / jnp.maximum(n, self.eps))

        y = self.drop(h, deterministic=not self.has_rng("dropout"))
        y = self.norm(y.reshape(B, self.num_heads, 1, head_dim))
        y = self.output_projection(y.reshape(B, self.hidden_dim))

        return ((c, n, m, h), buffer), y

    @nn.nowrap
    def initialize_carry(
        self,
        key: jax.Array,
        input_shape: Tuple[int, ...],
    ) -> Tuple:
        """Initialize the carry state.

        Args:
            key: Random key for initialization.
            input_shape: Shape of input (batch_size, features).

        Returns:
            Initial carry tuple of ((c, n, m, h), buffer).
        """
        *batch_dims, _ = input_shape
        carry_init = initializers.zeros_init()

        key_c, key_n, key_h, key_m, key_buf = random.split(key, 5)
        mem_shape = (*batch_dims, self.hidden_dim)

        c = carry_init(key_c, mem_shape, self.param_dtype)
        n = carry_init(key_n, mem_shape, self.param_dtype)
        m = carry_init(key_m, mem_shape, self.param_dtype)
        h = carry_init(key_h, mem_shape, self.param_dtype)

        buffer = carry_init(
            key_buf, (*batch_dims, self.conv_kernel_size, self.hidden_dim)
        )

        return (c, n, m, h), buffer

    @property
    def num_feature_axes(self) -> int:
        return 1
