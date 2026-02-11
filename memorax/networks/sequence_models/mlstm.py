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

from typing import Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.linen import initializers
from flax.typing import Dtype

from memorax.utils.typing import Array, Carry

from .memoroid import MemoroidCellBase
from .utils import BlockDiagonalDense, MultiHeadLayerNorm, wang_init


class mLSTMCell(MemoroidCellBase):
    """Matrix LSTM as a Memoroid algebra.

    Uses gated linear attention with matrix memory, computed efficiently
    via associative scan. Architecture follows NX-AI/xlstm.

    Element: (log_f, C, n, m) where:
        - log_f: cumulative log forget gate for relative decay
        - C: matrix memory contribution (k ⊗ v scaled by input gate)
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
    dtype: Dtype | None = None
    param_dtype: Dtype = jnp.float32

    def setup(self):
        if self.hidden_dim % self.num_heads != 0:
            raise ValueError(
                f"hidden_dim ({self.hidden_dim}) must be divisible by "
                f"num_heads ({self.num_heads})."
            )

        self.up_proj = nn.Dense(
            2 * self.hidden_dim,
            use_bias=False,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )

        self.conv = nn.Conv(
            features=self.hidden_dim,
            kernel_size=(self.conv_kernel_size,),
            padding=((self.conv_kernel_size - 1, 0),),
            feature_group_count=self.hidden_dim,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )

        self.q_proj = BlockDiagonalDense(
            self.hidden_dim,
            num_heads=self.num_heads,
            use_bias=False,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )
        self.k_proj = BlockDiagonalDense(
            self.hidden_dim,
            num_heads=self.num_heads,
            use_bias=False,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )
        self.v_proj = BlockDiagonalDense(
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
            bias_init=lambda key, shape, dtype=jnp.float32: jnp.linspace(3.0, 6.0, shape[0], dtype=dtype),
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

        self.out_proj = nn.Dense(
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

        up = self.up_proj(x)
        x_mlstm, z = jnp.split(up, 2, axis=-1)

        x_conv_act = nn.silu(self.conv(x_mlstm))

        q = self.q_proj(x_conv_act).reshape(B, T, self.num_heads, head_dim)
        k = self.k_proj(x_conv_act).reshape(B, T, self.num_heads, head_dim)
        v = self.v_proj(x_mlstm).reshape(B, T, self.num_heads, head_dim)

        return q, k, v, z, x_conv_act

    def __call__(self, x: Array, **kwargs) -> Carry:
        B, T, _ = x.shape
        head_dim = self.hidden_dim // self.num_heads

        q, k, v, _, _ = self._project(x)

        qkv = jnp.concatenate([q, k, v], axis=-1).reshape(B, T, -1)

        i_gate = self.i_gate(qkv).reshape(B, T, self.num_heads)
        f_gate = self.f_gate(qkv).reshape(B, T, self.num_heads)

        log_f = -jax.nn.softplus(-f_gate)
        log_i = i_gate

        m = log_i[:, :, :, None, None]
        i_stable = jnp.exp(log_i - m.squeeze(-1).squeeze(-1))

        k = k / jnp.sqrt(head_dim)

        k_col = k[:, :, :, :, None]
        v_row = v[:, :, :, None, :]
        kv_outer = k_col @ v_row

        i_expanded = i_stable[:, :, :, None, None]
        C = i_expanded * kv_outer

        n = i_stable[:, :, :, None] * k
        n = n[:, :, :, :, None]

        log_f = log_f[:, :, :, None, None]

        return (C, log_f, n, m)

    def binary_operator(self, a: Carry, b: Carry) -> Carry:
        C_a, log_f_a, n_a, m_a = a
        C_b, log_f_b, n_b, m_b = b

        log_f_combined = log_f_a + log_f_b

        m_a_decayed = m_a + log_f_b
        m_combined = jnp.maximum(m_a_decayed, m_b)

        scale_a = jnp.exp(m_a_decayed - m_combined)
        scale_b = jnp.exp(m_b - m_combined)

        C_combined = scale_a * C_a + scale_b * C_b
        n_combined = scale_a * n_a + scale_b * n_b

        return (C_combined, log_f_combined, n_combined, m_combined)

    def read(self, h: Carry, x: Array, **kwargs) -> Array:
        B, T, _ = x.shape
        head_dim = self.hidden_dim // self.num_heads

        q, _, _, z, x_conv_act = self._project(x)

        C, _, n, m = h

        q_row = q[:, :, :, None, :]

        qC = (q_row @ C).squeeze(-2)
        qn = (q_row @ n).squeeze(-2).squeeze(-1)

        max_val = jnp.exp(-m.squeeze(-1).squeeze(-1))
        normalizer = jnp.maximum(jnp.abs(qn), max_val)[:, :, :, None]
        h_tilde = qC / (normalizer + 1e-6)

        h_tilde = self.norm(
            h_tilde.transpose(0, 2, 1, 3)
        ).transpose(0, 2, 1, 3)
        h_tilde = h_tilde.reshape(B, T, self.hidden_dim)

        h_tilde = h_tilde + self.learnable_skip * x_conv_act

        y = h_tilde * nn.silu(z)

        y = self.out_proj(y)

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
        NH = self.num_heads
        head_dim = self.hidden_dim // NH
        H = NH * head_dim * head_dim

        # Unpack full previous carry
        C_acc, _, _, m_acc = carry
        # Unpack scan elements
        C_t, log_f_t, _, m_t = z

        # Compute scale factors (same as binary_operator)
        m_acc_decayed = m_acc + log_f_t
        m_new = jnp.maximum(m_acc_decayed, m_t)
        scale_old = jnp.exp(m_acc_decayed - m_new)
        scale_new = jnp.exp(m_t - m_new)

        # RTRL decay for C: ∂C_new/∂C_acc = scale_old (scalar per head)
        # Expand scale_old from (B,T,NH,1,1) to (B,T,NH*hd*hd)
        decay_flat = jnp.broadcast_to(
            scale_old, (B, T, NH, head_dim, head_dim)
        ).reshape(B, T, H)

        # Recompute f_gate_raw for ∂log_f/∂f_gate_bias
        q, k, v, _, _ = self._project(inputs)
        qkv = jnp.concatenate([q, k, v], axis=-1).reshape(B, T, -1)
        f_gate_raw = self.f_gate(qkv).reshape(B, T, NH)
        i_gate_raw = self.i_gate(qkv).reshape(B, T, NH)

        # ∂log_f/∂f_gate_bias[h] = σ(f_gate_raw[h])
        sigma_f = jax.nn.sigmoid(f_gate_raw)  # (B, T, NH)

        # indicator: m_acc + log_f >= m_t
        indicator = (m_acc_decayed >= m_t).squeeze(-1).squeeze(-1)  # (B, T, NH)

        # ∂C_new/∂log_f = scale_old*(1-ind)*C_acc - scale_new*ind*C_t
        # Both shapes (B,T,NH,hd,hd)
        dC_dlogf = (
            scale_old * (1.0 - indicator[:, :, :, None, None]) * C_acc
            - scale_new * indicator[:, :, :, None, None] * C_t
        )

        # J_f_bias[flat_h] = dC_dlogf[head_of(h)] * σ_f[head_of(h)]
        # Since both dC_dlogf and σ_f are per-head, the Jacobian per state element is:
        J_fgate = (dC_dlogf * sigma_f[:, :, :, None, None]).reshape(B, T, H)

        # For i_gate_bias: ∂C_new/∂i_gate_bias affects scale_new * C_t through log_i → m → scale
        # Simpler: ∂C_t/∂i_gate_bias since C_t = i_stable * kv_outer
        # i_stable = exp(log_i - m_t), log_i = i_gate_raw, m_t = log_i (from __call__)
        # So i_stable = exp(0) = 1 when m_t == log_i. ∂i_stable/∂i_gate_bias = 0 (self-normalizing)
        # But ∂scale_new/∂i_gate_bias through m_t = log_i:
        # ∂m_t/∂i_gate_bias = 1 (since m_t = log_i[:,:,:,None,None] = i_gate_raw[:,:,:,None,None])
        # ∂scale_new/∂i_gate_bias = -scale_new * ∂m_new/∂m_t * 1
        # ∂m_new/∂m_t = (1 - indicator)
        # Also scale_old changes through m_new
        dC_di = -scale_new * (1.0 - indicator[:, :, :, None, None]) * C_t
        J_igate = dC_di.reshape(B, T, H)

        return decay_flat, {
            "f_gate/bias": J_fgate,
            "i_gate/bias": J_igate,
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
        z = jnp.zeros((*batch_dims, 1, H))
        return {"f_gate/bias": z, "i_gate/bias": z}
