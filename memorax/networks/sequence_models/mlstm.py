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

from typing import Optional, Tuple

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.linen import initializers
from flax.typing import Dtype

from memorax.utils.typing import Array, Carry

from .memoroid import MemoroidCellBase
from .utils import BlockDiagonalDense, MultiHeadLayerNorm, wang_init


class mLSTMCell(MemoroidCellBase):
    """Matrix LSTM as a Memoroid algebra.

    Uses gated linear attention with matrix memory, computed efficiently
    via associative scan. Architecture follows NX-AI/xlstm.

    Element: (C, log_f, n, m) where:
        - C: matrix memory contribution (k ⊗ v scaled by input gate)
        - log_f: cumulative log forget gate for relative decay
        - n: normalizer contribution (k scaled by input gate)
        - m: max log value for numerical stability

    Combine: Accumulates states with relative exponential decay.

    Attributes:
        features: Output feature dimension.
        hidden_dim: Hidden dimension (inner embedding dim).
        num_heads: Number of attention heads.
        conv_kernel_size: Kernel size for causal convolution.
        dropout_rate: Dropout rate.
        dtype: Data type for computation.
        param_dtype: Data type for parameters.
    """

    features: int
    hidden_dim: int
    num_heads: int = 4
    conv_kernel_size: int = 4
    dropout_rate: float = 0.0
    dtype: Optional[Dtype] = None
    param_dtype: Dtype = jnp.float32

    def setup(self):
        if self.hidden_dim % self.num_heads != 0:
            raise ValueError(
                f"hidden_dim ({self.hidden_dim}) must be divisible by "
                f"num_heads ({self.num_heads})."
            )

        self.up_projection = nn.Dense(
            2 * self.hidden_dim,
            use_bias=False,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )

        self.causal_conv = nn.Conv(
            features=self.hidden_dim,
            kernel_size=(self.conv_kernel_size,),
            padding=((self.conv_kernel_size - 1, 0),),
            feature_group_count=self.hidden_dim,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )

        self.query = BlockDiagonalDense(
            self.hidden_dim,
            num_heads=self.num_heads,
            use_bias=False,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )
        self.key = BlockDiagonalDense(
            self.hidden_dim,
            num_heads=self.num_heads,
            use_bias=False,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )
        self.value = BlockDiagonalDense(
            self.hidden_dim,
            num_heads=self.num_heads,
            use_bias=False,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )

        self.i_gate = nn.Dense(
            self.num_heads,
            kernel_init=initializers.zeros_init(),
            bias_init=initializers.normal(stddev=0.1),
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )
        self.f_gate = nn.Dense(
            self.num_heads,
            kernel_init=initializers.zeros_init(),
            bias_init=lambda key, shape, dtype=jnp.float32: jnp.linspace(
                3.0, 6.0, shape[0], dtype=dtype
            ),
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )

        self.learnable_skip = self.param(
            "learnable_skip",
            nn.initializers.ones,
            (self.hidden_dim,),
        )

        self.norm = MultiHeadLayerNorm(
            use_scale=True,
            use_bias=False,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )

        self.output_projection = nn.Dense(
            self.features,
            use_bias=False,
            kernel_init=wang_init(self.hidden_dim, num_blocks=1),
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )

        self.drop = nn.Dropout(rate=self.dropout_rate)

    def _project(self, x: Array):
        B, T, _ = x.shape
        head_dim = self.hidden_dim // self.num_heads

        up = self.up_projection(x)
        x_mlstm, z = jnp.split(up, 2, axis=-1)

        u = nn.silu(self.causal_conv(x_mlstm))

        query = self.query(u).reshape(B, T, self.num_heads, head_dim)
        key = self.key(u).reshape(B, T, self.num_heads, head_dim)
        value = self.value(x_mlstm).reshape(B, T, self.num_heads, head_dim)

        return query, key, value, z, u

    def __call__(self, x: Array, **kwargs) -> Carry:
        B, T, _ = x.shape
        head_dim = self.hidden_dim // self.num_heads

        query, key, value, _, _ = self._project(x)

        gate_input = jnp.concatenate([query, key, value], axis=-1).reshape(B, T, -1)

        i_gate = self.i_gate(gate_input).reshape(B, T, self.num_heads)
        f_gate = self.f_gate(gate_input).reshape(B, T, self.num_heads)

        log_f = -jax.nn.softplus(-f_gate)
        log_i = i_gate

        # m = log_i makes exp(log_i - m) = 1 (self-normalizing)
        m = log_i[:, :, :, None, None]

        key = key / jnp.sqrt(head_dim)

        C = jnp.einsum("...i,...j->...ij", key, value)
        n = key[:, :, :, :, None]
        log_f = log_f[:, :, :, None, None]

        return (C, log_f, n, m)

    def binary_operator(self, a: Carry, b: Carry) -> Carry:
        C_a, log_f_a, n_a, m_a = a
        C_b, log_f_b, n_b, m_b = b

        log_f = log_f_a + log_f_b

        m_a_decayed = m_a + log_f_b
        m_combined = jnp.maximum(m_a_decayed, m_b)

        scale_a = jnp.exp(m_a_decayed - m_combined)
        scale_b = jnp.exp(m_b - m_combined)

        C = scale_a * C_a + scale_b * C_b
        n = scale_a * n_a + scale_b * n_b

        return (C, log_f, n, m_combined)

    def read(self, h: Carry, x: Array, **kwargs) -> Array:
        B, T, _ = x.shape

        query, _, _, z, u = self._project(x)

        C, _, n, m = h

        denominator = jnp.maximum(
            jnp.abs(jnp.einsum("...i,...i->...", query, n.squeeze(-1))),
            jnp.exp(-m.squeeze(-1).squeeze(-1)),
        )[:, :, :, None]
        h_tilde = jnp.einsum("...j,...jk->...k", query, C) / (denominator + 1e-6)

        h_tilde = self.norm(h_tilde.transpose(0, 2, 1, 3)).transpose(0, 2, 1, 3)
        h_tilde = h_tilde.reshape(B, T, self.hidden_dim)

        h_tilde = h_tilde + self.learnable_skip * u

        y = h_tilde * nn.silu(z)

        y = self.output_projection(y)

        y = self.drop(y, deterministic=not self.has_rng("dropout"))

        return y

    def initialize_carry(self, key: jax.Array, input_shape: Tuple[int, ...]) -> Carry:
        *batch_dims, _ = input_shape
        head_dim = self.hidden_dim // self.num_heads

        C = jnp.zeros((*batch_dims, 1, self.num_heads, head_dim, head_dim))
        log_f = jnp.zeros((*batch_dims, 1, self.num_heads, 1, 1))
        n = jnp.zeros((*batch_dims, 1, self.num_heads, head_dim, 1))
        m = jnp.full((*batch_dims, 1, self.num_heads, 1, 1), -1e9)

        return (C, log_f, n, m)

    def local_jacobian(self, carry, z, inputs, **kwargs):
        B, T = inputs.shape[:2]
        num_heads = self.num_heads
        head_dim = self.hidden_dim // num_heads
        H = num_heads * head_dim * head_dim

        C, _, _, m = carry
        C_t, log_f_t, _, m_t = z

        # Scale factors (same logic as binary_operator)
        m_decayed = m + log_f_t
        m = jnp.maximum(m_decayed, m_t)
        scale_a = jnp.exp(m_decayed - m)
        scale_b = jnp.exp(m_t - m)

        # RTRL decay for C: ∂C/∂C_prev = scale_a (scalar per head)
        decay_flat = jnp.broadcast_to(
            scale_a, (B, T, num_heads, head_dim, head_dim)
        ).reshape(B, T, H)

        # Recompute gates for Jacobians
        query, key, value, _, _ = self._project(inputs)
        gate_input = jnp.concatenate([query, key, value], axis=-1).reshape(B, T, -1)
        f_gate = self.f_gate(gate_input).reshape(B, T, num_heads)

        # Expand per-head scalars to 5D for broadcasting
        sigma_f = jax.nn.sigmoid(f_gate)[:, :, :, None, None]
        indicator = m_decayed >= m_t  # already (B,T,NH,1,1)

        # ∂C/∂log_f = scale_a*(1-ind)*C - scale_b*ind*C_t
        dC_dlogf = scale_a * (1.0 - indicator) * C - scale_b * indicator * C_t
        J_f_gate = (dC_dlogf * sigma_f).reshape(B, T, H)

        # ∂C/∂i_gate through m_t = log_i (self-normalizing)
        dC_di = -scale_b * (1.0 - indicator) * C_t
        J_i_gate = dC_di.reshape(B, T, H)

        return decay_flat, {
            "f_gate/bias": J_f_gate,
            "i_gate/bias": J_i_gate,
        }

    def get_param_indices(self):
        head_dim = self.hidden_dim // self.num_heads
        H = self.num_heads * head_dim * head_dim
        idx = jnp.arange(H) // (head_dim * head_dim)
        return {"f_gate/bias": idx, "i_gate/bias": idx}

    def initialize_sensitivity(self, key, input_shape):
        *batch_dims, _ = input_shape
        head_dim = self.hidden_dim // self.num_heads
        H = self.num_heads * head_dim * head_dim
        zeros = jnp.zeros((*batch_dims, 1, H))
        return {"f_gate/bias": zeros, "i_gate/bias": zeros}
