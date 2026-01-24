from functools import partial
from typing import Tuple

import jax.numpy as jnp
from flax import linen as nn
from flax.typing import Dtype, Initializer

from memorax.utils.typing import Array, Carry

from .memoroid import MemoroidCellBase


class MambaCell(MemoroidCellBase):
    """Mamba selective SSM as a memoroid algebra.

    Implements the Mamba architecture from "Mamba: Linear-Time Sequence Modeling
    with Selective State Spaces" (Gu & Dao, 2023).

    Uses input-dependent dynamics (dt, B, C) for content-aware state updates.
    Projections are handled internally for a clean API.

    Element: (decay, state)
    Combine: (a_j * a_i, a_j * s_i + s_j)

    Args:
        features: Input/output feature dimension.
        num_heads: Number of attention heads.
        head_dim: Dimension per head.
        hidden_dim: SSM state dimension.
        expansion_factor: Expansion factor for internal projection.
        kernel_init: Kernel initializer.
        bias_init: Bias initializer.
        dtype: Computation dtype.
        param_dtype: Parameter dtype.
    """

    features: int
    num_heads: int = 8
    head_dim: int = 16
    hidden_dim: int = 16
    expansion_factor: int = 2
    kernel_init: Initializer = nn.initializers.lecun_normal()
    bias_init: Initializer = nn.initializers.zeros_init()
    dtype: Dtype = jnp.float32
    param_dtype: Dtype = jnp.float32

    def setup(self):
        self.log_decay = self.param(
            "log_decay", nn.initializers.normal(stddev=0.1), (self.num_heads,)
        )
        self.skip_weight = self.param(
            "skip_weight", nn.initializers.ones, (self.num_heads, self.head_dim)
        )

    @nn.compact
    def __call__(self, x: Array, **kwargs) -> Carry:
        """Compute Mamba elements with internal projections.

        Args:
            x: Input of shape (B, T, features)

        Returns:
            Carry tuple of (decay, state, gate, x_proj) for binary_operator and read.
        """
        batch_size, seq_len, _ = x.shape
        inner_dim = self.num_heads * self.head_dim

        # Input projection with expansion for gating
        x_proj = nn.Dense(
            inner_dim * self.expansion_factor,
            kernel_init=self.kernel_init,
            use_bias=False,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="in_proj",
        )(x)

        # Split into x and gate
        x_inner, gate = jnp.split(x_proj, 2, axis=-1)
        gate = nn.silu(gate)

        # Reshape x to (B, T, num_heads, head_dim)
        x_inner = x_inner.reshape(batch_size, seq_len, self.num_heads, self.head_dim)

        # Project to dt, B, C
        dt = nn.Dense(
            self.num_heads,
            kernel_init=self.kernel_init,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="dt_proj",
        )(x)
        dt = nn.softplus(dt)

        B = nn.Dense(
            self.num_heads * self.hidden_dim,
            kernel_init=self.kernel_init,
            use_bias=False,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="B_proj",
        )(x)
        B = B.reshape(batch_size, seq_len, self.num_heads, self.hidden_dim)

        # Compute decay and state
        decay_rate = -jnp.exp(self.log_decay)
        decay = jnp.exp(dt * decay_rate[None, None, :])
        decay = decay[:, :, :, None, None]

        # State: outer product of B and x_inner, scaled by dt
        state = jnp.einsum("bthn,bthd->bthnd", B * dt[:, :, :, None], x_inner)

        # Return carry with gate and x_inner for read()
        return (decay, state, gate, x_inner)

    def binary_operator(self, a: Carry, b: Carry) -> Carry:
        """Diagonal SSM combine: (a_j * a_i, a_j * s_i + s_j)"""
        decay_i, state_i, gate_i, x_i = a
        decay_j, state_j, gate_j, x_j = b
        return (
            decay_j * decay_i,
            decay_j * state_i + state_j,
            gate_j,  # Keep latest gate
            x_j,  # Keep latest x
        )

    @nn.compact
    def read(self, h: Carry, x: Array, **kwargs) -> Array:
        """Compute output from accumulated state.

        Args:
            h: Accumulated state (decay, state, gate, x_inner)
            x: Original input of shape (B, T, features)

        Returns:
            Output of shape (B, T, features)
        """
        batch_size, seq_len, _ = x.shape
        _, state, gate, x_inner = h

        # Project to C
        C = nn.Dense(
            self.num_heads * self.hidden_dim,
            kernel_init=self.kernel_init,
            use_bias=False,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="C_proj",
        )(x)
        C = C.reshape(batch_size, seq_len, self.num_heads, self.hidden_dim)

        # Compute output
        output = jnp.einsum("bthn,bthnd->bthd", C, state)
        output = output + self.skip_weight[None, None, :, :] * x_inner

        # Reshape and apply gate
        output = output.reshape(batch_size, seq_len, self.num_heads * self.head_dim)
        output = output * gate

        # Output projection
        output = nn.Dense(
            self.features,
            kernel_init=self.kernel_init,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="out_proj",
        )(output)

        return output

    def initialize_carry(self, key, input_shape: Tuple[int, ...]) -> Carry:
        batch_size, *_ = input_shape
        inner_dim = self.num_heads * self.head_dim
        decay = jnp.ones(
            (batch_size, 1, self.num_heads, 1, 1),
            dtype=self.dtype,
        )
        state = jnp.zeros(
            (batch_size, 1, self.num_heads, self.hidden_dim, self.head_dim),
            dtype=self.dtype,
        )
        gate = jnp.ones(
            (batch_size, 1, inner_dim),
            dtype=self.dtype,
        )
        x_inner = jnp.zeros(
            (batch_size, 1, self.num_heads, self.head_dim),
            dtype=self.dtype,
        )
        return (decay, state, gate, x_inner)
