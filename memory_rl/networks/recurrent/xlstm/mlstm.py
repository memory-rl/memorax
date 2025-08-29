from typing import Tuple

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp

from .utils import BlockLinear, CausalConv1D

from functools import partial  # pylint: disable=g-importing-member
from typing import (
    Any,
    TypeVar,
)
from collections.abc import Callable

import jax
from jax import numpy as jnp
from jax import random

from flax.linen import initializers
from flax.linen.activation import sigmoid
from flax.linen.linear import Dense, default_kernel_init
from flax.linen.module import Module, compact, nowrap
from flax.linen.recurrent import RNNCellBase
from flax.typing import (
    PRNGKey,
    Dtype,
    Initializer,
)

A = TypeVar("A")
Carry = Any
CarryHistory = Any
Output = Any


class mLSTMCell(RNNCellBase):
    r"""mLSTM cell (xLSTM), matrix memory with stabilized exponential gates.

    Equations (single head), cf. xLSTM:
      C_t = f_t * C_{t-1} + i_t * v_t k_t^T
      n_t = f_t * n_{t-1} + i_t * k_t
      h_t = o_t ⊙ (C_t q_t / max(|n_t^T q_t|, 1))
      q_t = W_q x_t + b_q
      k_t = (1/√d) W_k x_t + b_k
      v_t = W_v x_t + b_v
      i_t = exp(î_t) , î_t = w_i^T x_t + b_i
      f_t = σ(f̂_t) OR exp(f̂_t) , f̂_t = w_f^T x_t + b_f
      o_t = σ(ô_t) , ô_t = W_o x_t + b_o

    Stabilization (per-head) as in Eq. (15):
      m_t = max(log f_t + m_{t-1}, log i_t)
      i'_t = exp(log i_t - m_t) = exp(î_t - m_t)
      f'_t = exp(log f_t + m_{t-1} - m_t)

    Carry = (C, n, m, h), where:
      C: (..., num_heads, d_h, d_h)
      n: (..., num_heads, d_h)
      m: (..., num_heads)  (stabilizer)
      h: (..., features)

    Attributes:
      features: total model dimension (num_heads * head_dim).
      num_heads: number of independent heads (no mixing).
      forget_activation: 'exp' or 'sigmoid' for f_t.
      kernel_init, bias_init, dtype, param_dtype, carry_init: as in LSTMCell.
      normalizer_threshold: lower bound for |n^T q| (default 1.0).
    """

    features: int
    num_heads: int = 1
    forget_activation: str = "exp"  # 'exp' (paper's default) or 'sigmoid'
    kernel_init: Initializer = default_kernel_init
    bias_init: Initializer = initializers.zeros_init()
    dtype: Dtype | None = None
    param_dtype: Dtype = jnp.float32
    carry_init: Initializer = initializers.zeros_init()
    normalizer_threshold: float = 1.0

    @compact
    def __call__(self, carry, inputs):
        (C, n, m, h_prev) = carry
        assert (
            self.features % self.num_heads == 0
        ), "features must be divisible by num_heads"
        d_h = self.features // self.num_heads
        batch_dims = inputs.shape[:-1]

        dense_vec = partial(
            Dense,
            use_bias=True,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )

        # Projections: q, k, v in R^{..., H, d_h}; o in R^{..., H, d_h} (flattened to features).
        q = dense_vec(name="Wq", features=self.features)(inputs)
        k = dense_vec(name="Wk", features=self.features)(inputs)
        v = dense_vec(name="Wv", features=self.features)(inputs)
        o_lin = dense_vec(name="Wo", features=self.features)(inputs)

        # Reshape to heads.
        def to_heads(x):
            return x.reshape(x.shape[:-1] + (self.num_heads, d_h))

        q = to_heads(q)
        k = to_heads(k) / jnp.sqrt(
            jnp.array(d_h, dtype=self.param_dtype)
        )  # (1/√d) scaling as in paper
        v = to_heads(v)
        o = sigmoid(to_heads(o_lin))  # elementwise output gate

        # Scalar gates per head: î, f̂ in R^{..., H}
        gate_dense = partial(
            Dense,
            features=self.num_heads,
            use_bias=True,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )
        i_hat = gate_dense(name="wi")(inputs)  # î_t
        f_hat = gate_dense(name="wf")(inputs)  # f̂_t

        # log(i_t) = î_t (since i_t = exp(î_t))
        log_it = i_hat

        # log(f_t): depends on chosen activation
        if self.forget_activation == "exp":
            log_ft = f_hat
        elif self.forget_activation == "sigmoid":
            # log σ(x) = -softplus(-x)
            log_ft = -jax.nn.softplus(-f_hat)
        else:
            raise ValueError("forget_activation must be 'exp' or 'sigmoid'")

        # Stabilization state m_t and stabilized gates i', f' (all per-head)
        m_t = jnp.maximum(log_ft + m, log_it)  # m_t = max(log f_t + m_{t-1}, log i_t)
        i_stab = jnp.exp(log_it - m_t)  # i'_t
        f_stab = jnp.exp(log_ft + m - m_t)  # f'_t

        # Update matrix memory C and normalizer n
        # Expand gates for broadcasting
        i_b = i_stab[..., None, None]  # (..., H, 1, 1)
        f_b = f_stab[..., None, None]  # (..., H, 1, 1)
        C_new = f_b * C + i_b * (v[..., :, None] * k[..., None, :])  # v k^T

        i_nv = i_stab[..., None]  # (..., H, 1)
        f_nv = f_stab[..., None]  # (..., H, 1)
        n_new = f_nv * n + i_nv * k

        # Compute hidden pre-activation: C_t q_t
        Cq = jnp.matmul(C_new, q[..., :, None])[..., :, 0]  # (..., H, d_h)

        # Denominator: max(|n^T q|, threshold)
        denom = jnp.maximum(
            jnp.abs(jnp.sum(n_new * q, axis=-1)),
            jnp.asarray(self.normalizer_threshold, dtype=Cq.dtype),
        )  # (..., H)
        h_tilde = Cq / denom[..., None]  # (..., H, d_h)

        # Output gate
        h_new_heads = o * h_tilde
        h_new = h_new_heads.reshape(batch_dims + (self.features,))

        return (C_new, n_new, m_t, h_new), h_new

    @nowrap
    def initialize_carry(self, rng: PRNGKey, input_shape: tuple[int, ...]):
        batch_dims = input_shape[:-1]
        assert (
            self.features % self.num_heads == 0
        ), "features must be divisible by num_heads"
        d_h = self.features // self.num_heads
        key_C, key_n, key_m, key_h = random.split(rng, 4)
        C = self.carry_init(
            key_C, batch_dims + (self.num_heads, d_h, d_h), self.param_dtype
        )
        n = self.carry_init(key_n, batch_dims + (self.num_heads, d_h), self.param_dtype)
        m = self.carry_init(key_m, batch_dims + (self.num_heads,), self.param_dtype)
        h = self.carry_init(key_h, batch_dims + (self.features,), self.param_dtype)
        return (C, n, m, h)

    @property
    def num_feature_axes(self) -> int:
        return 1


class mLSTMBlock(RNNCellBase):
    r"""mLSTM Block (Figure 11) — pre up-projection residual block.

    Carry = (C, n, m, conv_buf), where:
      C: (..., H, d_h, d_h)   matrix memory
      n: (..., H, d_h)        normalizer
      m: (..., H)             stabilizer (log-scale)
      conv_buf: (..., D_up, K-1) causal buffer for depthwise conv (per-channel)
    Output:
      y in base dimension D (residual: y = x + down_proj(...))

    Key details matched to paper:
      • Pre-LN residual block; two PF=2 up-projections; PF=½ down-projection.  (Fig. 11)
      • Depthwise (dimension-wise) causal Conv1D with kernel size 4, Swish activation. (Fig. 11)
      • q, k via block-diagonal projections (block size 4); v bypasses conv (still block-diagonal). (Fig. 11)
      • mLSTM core with exponential gating + stabilization (Eq. 15) and 1/√d scaling on k. (Sec. 2.3)
      • External, component-wise output gate using Swish (outside the cell). (Fig. 11)
      • GroupNorm = head-wise LayerNorm (per-head LN). (Fig. 11)

    Attributes:
      features: base model dim D.
      num_heads: H (default 4 as shown).
      up_proj_factor: PF for up-projection (default 2).
      conv1d_kernel_size: depthwise causal kernel (default 4 as shown).
      qkv_proj_blocksize: block size for block-diagonal projections (default 4 as shown).
      forget_activation: 'exp' or 'sigmoid' for f_t inside mLSTM (default 'exp').
      normalizer_threshold: clamp for |n^T q| (default 1.0).
      dtype / param_dtype / kernel_inits / bias_init / carry_init: like your LSTMCell.
    """

    features: int
    num_heads: int = 4
    up_proj_factor: int = 2
    conv1d_kernel_size: int = 4
    qkv_proj_blocksize: int = 4
    forget_activation: str = "exp"
    normalizer_threshold: float = 1.0

    kernel_init: Initializer = default_kernel_init
    bias_init: Initializer = initializers.zeros_init()
    dtype: Dtype | None = None
    param_dtype: Dtype = jnp.float32
    carry_init: Initializer = initializers.zeros_init()
    ln_epsilon: float = 1e-5
    gn_epsilon: float = 1e-5

    def _layer_norm(self, x, name: str):
        d = x.shape[-1]
        scale = self.param(f"{name}_scale", self.bias_init, (d,), self.param_dtype)
        bias = self.param(f"{name}_bias", self.bias_init, (d,), self.param_dtype)
        mean = jnp.mean(x, axis=-1, keepdims=True)
        var = jnp.mean((x - mean) ** 2, axis=-1, keepdims=True)
        xhat = (x - mean) / jnp.sqrt(var + self.ln_epsilon)
        return xhat * (scale + 1.0) + bias

    def _groupnorm_heads(self, x_hd, name: str):
        H, d_h = x_hd.shape[-2], x_hd.shape[-1]
        scale = self.param(f"{name}_scale", self.bias_init, (H, d_h), self.param_dtype)
        bias = self.param(f"{name}_bias", self.bias_init, (H, d_h), self.param_dtype)
        mean = jnp.mean(x_hd, axis=-1, keepdims=True)
        var = jnp.mean((x_hd - mean) ** 2, axis=-1, keepdims=True)
        xhat = (x_hd - mean) / jnp.sqrt(var + self.gn_epsilon)
        return xhat * (scale + 1.0) + bias

    def _block_diag_proj(self, x, blocksize: int, name: str):
        # Block-diagonal linear map with square blocks of size `blocksize`.
        d = x.shape[-1]
        assert d % blocksize == 0, "channel dim must be divisible by blocksize"
        n_blocks = d // blocksize

        W = self.param(
            f"{name}_W",
            self.kernel_init,
            (n_blocks, blocksize, blocksize),
            self.param_dtype,
        )
        b = self.param(
            f"{name}_b", self.bias_init, (n_blocks, blocksize), self.param_dtype
        )

        # reshape to (..., G, C) where G=n_blocks, C=blocksize
        xg = x.reshape(x.shape[:-1] + (n_blocks, blocksize))

        # einsum over a single block index 'g': (..., g, c) @ (g, c, d) -> (..., g, d)
        yg = jnp.einsum("...gc,gcd->...gd", xg, W) + b

        return yg.reshape(x.shape[:-1] + (d,))

    def _depthwise_causal_conv1d(self, x_t, buf, name: str):
        K = self.conv1d_kernel_size
        d = x_t.shape[-1]
        w = self.param(f"{name}_W", self.kernel_init, (d, K), self.param_dtype)
        b = self.param(f"{name}_b", self.bias_init, (d,), self.param_dtype)
        # buf[..., i] stores x_{t-1-i}
        acc = w[:, 0] * x_t
        for j in range(1, K):
            acc = acc + w[:, j] * buf[..., j - 1]
        y = acc + b
        # update buffer: new front is x_t
        new_buf = (
            jnp.concatenate([x_t[..., None], buf[..., : K - 2]], axis=-1)
            if K > 1
            else buf
        )
        return y, new_buf

    def _mlstm_core(self, carry_core, q, k, v, gate_src):
        C, n, m = carry_core
        H, d_h = C.shape[-3], C.shape[-2]

        # Heads reshape
        def to_heads(z):
            return z.reshape(z.shape[:-1] + (H, d_h))

        qh, kh, vh = map(to_heads, (q, k, v))
        kh = kh / jnp.sqrt(jnp.asarray(d_h, dtype=self.param_dtype))

        # gates i, f from gate_src (conv’d path), logits per head
        dense_gate = partial(
            Dense,
            features=H,
            use_bias=True,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )
        i_hat = dense_gate(name="core_wi")(gate_src)  # î_t
        f_hat = dense_gate(name="core_wf")(gate_src)  # f̂_t

        log_it = i_hat
        if self.forget_activation == "exp":
            log_ft = f_hat
        elif self.forget_activation == "sigmoid":
            log_ft = -jax.nn.softplus(-f_hat)  # log σ(x)
        else:
            raise ValueError("forget_activation must be 'exp' or 'sigmoid'")

        # stabilization (Eq. 15)
        m_t = jnp.maximum(log_ft + m, log_it)
        i_stab = jnp.exp(log_it - m_t)
        f_stab = jnp.exp(log_ft + m - m_t)

        # updates
        i_b, f_b = i_stab[..., None, None], f_stab[..., None, None]  # for C
        C_new = f_b * C + i_b * (vh[..., :, None] * kh[..., None, :])

        i_nv, f_nv = i_stab[..., None], f_stab[..., None]  # for n
        n_new = f_nv * n + i_nv * kh

        # read
        Cq = jnp.matmul(C_new, qh[..., :, None])[..., :, 0]  # (..., H, d_h)
        denom = jnp.maximum(
            jnp.abs(jnp.sum(n_new * qh, axis=-1)),
            jnp.asarray(self.normalizer_threshold, dtype=Cq.dtype),
        )
        h_hd = Cq / denom[..., None]  # (..., H, d_h); no internal o_t
        return (C_new, n_new, m_t), h_hd

    @compact
    def __call__(self, carry, inputs):
        (C, n, m, conv_buf) = carry
        D = self.features
        D_up = self.up_proj_factor * D
        assert (
            D_up % self.num_heads == 0
        ), "up-projected dim must be divisible by num_heads"
        assert (
            D_up % self.qkv_proj_blocksize == 0
        ), "up-projected dim must be divisible by blocksize"

        # Pre-LN
        x = inputs
        x_ln = self._layer_norm(x, name="pre_ln")

        dense_up = partial(
            Dense,
            features=D_up,
            use_bias=True,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )
        # Two PF=2 branches
        up_gate = dense_up(name="up_gate")(x_ln)  # external gate branch (PF=2)
        up_core = dense_up(name="up_core")(x_ln)  # mLSTM core branch  (PF=2)

        # depthwise causal conv (k=4) on core branch + Swish
        conv_out, conv_buf_new = self._depthwise_causal_conv1d(
            up_core, conv_buf, name="dwconv"
        )
        conv_act = jax.nn.swish(conv_out)

        # q,k from conv’d path (block-diagonal); v from *pre-conv* path (also block-diagonal)
        q = self._block_diag_proj(conv_act, self.qkv_proj_blocksize, name="q_bd")
        k = self._block_diag_proj(conv_act, self.qkv_proj_blocksize, name="k_bd")
        v = self._block_diag_proj(
            up_core, self.qkv_proj_blocksize, name="v_bd"
        )  # conv bypass (Fig. 11)

        # mLSTM core (no internal output gate), gating source = conv’d path
        (C_new, n_new, m_new), h_hd = self._mlstm_core(
            (C, n, m), q, k, v, gate_src=conv_act
        )

        # GroupNorm (head-wise LN), then add learnable skip from conv’d path
        h_up = h_hd.reshape(h_hd.shape[:-2] + (D_up,))
        h_hd_norm = self._groupnorm_heads(h_hd, name="gn")
        h_norm_up = h_hd_norm.reshape(h_up.shape)

        lskip = Dense(
            features=D_up,
            use_bias=True,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="lskip",
        )(conv_act)

        # External component-wise gate (Swish) and down-projection (PF=½)
        gate = jax.nn.swish(up_gate)
        gated = gate * (h_norm_up + lskip)

        y = Dense(
            features=D,
            use_bias=True,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="down_proj",
        )(gated)

        out = x + y  # pre-LN residual

        return (C_new, n_new, m_new, conv_buf_new), out

    @nowrap
    def initialize_carry(self, rng: PRNGKey, input_shape: tuple[int, ...]):
        batch_dims = input_shape[:-1]
        D = self.features
        D_up = self.up_proj_factor * D
        assert (
            D_up % self.num_heads == 0
        ), "up-projected dim must be divisible by num_heads"
        d_h = D_up // self.num_heads
        keyC, keyn, keym, keybuf = random.split(rng, 4)
        C = self.carry_init(
            keyC, batch_dims + (self.num_heads, d_h, d_h), self.param_dtype
        )
        n = self.carry_init(keyn, batch_dims + (self.num_heads, d_h), self.param_dtype)
        m = self.carry_init(keym, batch_dims + (self.num_heads,), self.param_dtype)
        conv_buf = jnp.zeros(
            batch_dims + (D_up, max(self.conv1d_kernel_size - 1, 0)), self.param_dtype
        )
        return (C, n, m, conv_buf)

    @property
    def num_feature_axes(self) -> int:
        return 1
