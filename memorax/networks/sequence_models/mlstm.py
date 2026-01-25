"""mLSTM as a Memoroid algebra for efficient parallel computation.

The mLSTM (matrix LSTM) is a linear attention variant with learned
per-step gating. By formulating it as a Memoroid algebra, we can
use associative scan for O(log n) parallel depth instead of O(n)
sequential RNN computation.

Core recurrence:
    C_new = f * C + i * (k ⊗ v)   # matrix memory
    n_new = f * n + i * k          # normalizer
    output = (q @ C) / (q @ n)     # query

This is associative when we track cumulative decay properly.
"""

from functools import partial
from typing import Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.linen import initializers
from flax.linen.linear import Dense
from flax.typing import Dtype

from memorax.utils.typing import Array, Carry

from .memoroid import MemoroidCellBase
from .utils import wang_init


class mLSTMCell(MemoroidCellBase):
    """Matrix LSTM as a Memoroid algebra.

    Uses gated linear attention with matrix memory, computed efficiently
    via associative scan.

    Element: (log_f, C, n, m) where:
        - log_f: cumulative log forget gate for relative decay
        - C: matrix memory contribution (k ⊗ v scaled by input gate)
        - n: normalizer contribution (k scaled by input gate)
        - m: max log value for numerical stability

    Combine: Accumulates states with relative exponential decay.

    Attributes:
        features: Output feature dimension.
        hidden_dim: Hidden dimension (before expansion).
        num_heads: Number of attention heads.
        head_dim: Dimension per head (computed as hidden_dim / num_heads).
        dropout_rate: Dropout rate.
        dtype: Data type for computation.
        param_dtype: Data type for parameters.
    """

    features: int
    hidden_dim: int
    num_heads: int = 4
    dropout_rate: float = 0.0
    dtype: Dtype | None = None
    param_dtype: Dtype = jnp.float32

    @nn.compact
    def __call__(self, x: Array, **kwargs) -> Carry:
        """Compute mLSTM elements for parallel scan.

        Args:
            x: Input of shape (B, T, D)

        Returns:
            Carry tuple of (log_f, C, n, m) where:
                - log_f: (B, T, NH, 1, 1) cumulative log forget
                - C: (B, T, NH, DH, DH) matrix memory contribution
                - n: (B, T, NH, DH, 1) normalizer contribution
                - m: (B, T, NH, 1, 1) max log value for stability
        """
        B, T, _ = x.shape
        head_dim = self.hidden_dim // self.num_heads

        if self.hidden_dim % self.num_heads != 0:
            raise ValueError(
                f"hidden_dim ({self.hidden_dim}) must be divisible by "
                f"num_heads ({self.num_heads})."
            )

        # Project to hidden dimension
        x_proj = Dense(
            features=self.hidden_dim,
            use_bias=False,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="in_proj",
        )(x)

        # Q, K, V projections
        projection = partial(
            Dense,
            features=self.hidden_dim,
            use_bias=False,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )

        q = projection(name="q")(x_proj)
        k = projection(name="k")(x_proj)
        v = projection(name="v")(x_proj)

        # Reshape to heads: (B, T, NH, DH)
        q = q.reshape(B, T, self.num_heads, head_dim)
        k = k.reshape(B, T, self.num_heads, head_dim)
        v = v.reshape(B, T, self.num_heads, head_dim)

        # Gate projections from concatenated qkv
        qkv = jnp.concatenate([q, k, v], axis=-1)  # (B, T, NH, 3*DH)
        qkv_flat = qkv.reshape(B, T, -1)  # (B, T, NH*3*DH)

        gate = partial(
            Dense,
            features=self.num_heads,
            kernel_init=initializers.zeros_init(),
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )

        i_gate = gate(name="wi", bias_init=initializers.normal(stddev=0.1))(qkv_flat)
        f_gate = gate(name="wf", bias_init=initializers.constant(4.0))(qkv_flat)

        # Reshape gates: (B, T, NH)
        i_gate = i_gate.reshape(B, T, self.num_heads)
        f_gate = f_gate.reshape(B, T, self.num_heads)

        # Log-space gates for numerical stability
        log_f = -jax.nn.softplus(-f_gate)  # (B, T, NH)
        log_i = i_gate  # Keep in log space

        # Compute max for stability: m = max(log_i)
        m = log_i[:, :, :, None, None]  # (B, T, NH, 1, 1)

        # Stable input gate
        i_stable = jnp.exp(log_i - m.squeeze(-1).squeeze(-1))  # (B, T, NH)

        # Scale k and v by sqrt for numerical stability
        k = k / jnp.sqrt(head_dim)

        # Compute contributions
        # C_contribution = i * (k ⊗ v): (B, T, NH, DH, DH)
        k_col = k[:, :, :, :, None]  # (B, T, NH, DH, 1)
        v_row = v[:, :, :, None, :]  # (B, T, NH, 1, DH)
        kv_outer = k_col @ v_row  # (B, T, NH, DH, DH)

        i_expanded = i_stable[:, :, :, None, None]  # (B, T, NH, 1, 1)
        C = i_expanded * kv_outer  # (B, T, NH, DH, DH)

        # n_contribution = i * k: (B, T, NH, DH, 1)
        n = i_stable[:, :, :, None] * k  # (B, T, NH, DH)
        n = n[:, :, :, :, None]  # (B, T, NH, DH, 1)

        # Reshape log_f for broadcasting: (B, T, NH, 1, 1)
        log_f = log_f[:, :, :, None, None]

        return (log_f, C, n, m)

    def binary_operator(self, a: Carry, b: Carry) -> Carry:
        """Combine two elements with decay-weighted accumulation.

        When combining element a (earlier positions) with element b
        (later positions), a's state decays by b's cumulative forget gate.

        Args:
            a: Earlier element (log_f_a, C_a, n_a, m_a)
            b: Later element (log_f_b, C_b, n_b, m_b)

        Returns:
            Combined element with accumulated state.
        """
        log_f_a, C_a, n_a, m_a = a
        log_f_b, C_b, n_b, m_b = b

        # Combined cumulative decay
        log_f_combined = log_f_a + log_f_b

        # Numerical stability: find max of decayed a and b
        m_a_decayed = m_a + log_f_b  # a's contribution after decay
        m_combined = jnp.maximum(m_a_decayed, m_b)

        # Compute stable scaling factors
        scale_a = jnp.exp(m_a_decayed - m_combined)
        scale_b = jnp.exp(m_b - m_combined)

        # Combine states
        C_combined = scale_a * C_a + scale_b * C_b
        n_combined = scale_a * n_a + scale_b * n_b

        return (log_f_combined, C_combined, n_combined, m_combined)

    @nn.compact
    def read(self, h: Carry, x: Array, **kwargs) -> Array:
        """Query accumulated memory to produce output.

        Args:
            h: Accumulated state (log_f, C, n, m)
            x: Original input of shape (B, T, D)

        Returns:
            Output of shape (B, T, features)
        """
        B, T, in_features = x.shape
        head_dim = self.hidden_dim // self.num_heads

        # Project input for query
        x_proj = Dense(
            features=self.hidden_dim,
            use_bias=False,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="in_proj",
        )(x)

        q = Dense(
            features=self.hidden_dim,
            use_bias=False,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="q",
        )(x_proj)

        # Reshape to heads: (B, T, NH, DH)
        q = q.reshape(B, T, self.num_heads, head_dim)
        q = q / jnp.sqrt(head_dim)

        # Extract accumulated state
        _, C, n, m = h

        # Query the memory: h_tilde = (q @ C) / normalize(q @ n)
        q_row = q[:, :, :, None, :]  # (B, T, NH, 1, DH)

        # q @ C: (B, T, NH, 1, DH) @ (B, T, NH, DH, DH) -> (B, T, NH, 1, DH)
        qC = (q_row @ C).squeeze(-2)  # (B, T, NH, DH)

        # q @ n: (B, T, NH, 1, DH) @ (B, T, NH, DH, 1) -> (B, T, NH, 1, 1)
        qn = (q_row @ n).squeeze(-2).squeeze(-1)  # (B, T, NH)

        # Normalize
        normalizer = jnp.maximum(jnp.abs(qn), 1.0)[:, :, :, None]
        h_tilde = qC / (normalizer + 1e-6)

        # Reshape and project output
        h_tilde = h_tilde.reshape(B, T, self.hidden_dim)

        # Output projection
        y = Dense(
            features=self.features,
            use_bias=False,
            kernel_init=wang_init(self.hidden_dim, num_blocks=1),
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="out_proj",
        )(h_tilde)

        # Dropout
        y = nn.Dropout(
            rate=self.dropout_rate, deterministic=not self.has_rng("dropout")
        )(y)

        return y

    def initialize_carry(self, key: jax.Array, input_shape: Tuple[int, ...]) -> Carry:
        """Initialize carry with identity decay and zero state.

        Args:
            key: Random key (unused).
            input_shape: Shape of input (*batch_dims, features).

        Returns:
            Initial carry tuple (log_f, C, n, m).
        """
        *batch_dims, _ = input_shape
        head_dim = self.hidden_dim // self.num_heads

        # Initial cumulative log decay is 0 (no decay yet)
        log_f = jnp.zeros((*batch_dims, 1, self.num_heads, 1, 1))

        # Initial state is zero
        C = jnp.zeros((*batch_dims, 1, self.num_heads, head_dim, head_dim))
        n = jnp.zeros((*batch_dims, 1, self.num_heads, head_dim, 1))

        # Initial max is -inf (will be overwritten on first combine)
        m = jnp.full((*batch_dims, 1, self.num_heads, 1, 1), -1e9)

        return (log_f, C, n, m)
