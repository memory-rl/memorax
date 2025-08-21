from dataclasses import dataclass
from typing import Any, Optional

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import jax.lax as lax


# ---------------------------
# Config
# ---------------------------
@dataclass(frozen=True)
class GTrXLConfig:
    mem_len: int
    num_layers: int
    num_heads: int
    num_embeds: int
    dropout_rate: float = 0.1
    use_bias: bool = True
    dtype: Any = jnp.float32


# ---------------------------
# Utilities
# ---------------------------
def _sinusoidal_positions(n_pos: int, d: int, dtype=jnp.float32):
    """Return (n_pos, d) sinusoidal embeddings for relative positions 0..n_pos-1."""
    # Standard Transformer sinusoidal table
    half = d // 2
    freqs = 1.0 / (10000 ** (jnp.arange(0, half, dtype=dtype) / float(half)))
    pos = jnp.arange(n_pos, dtype=dtype)[:, None]  # (n_pos, 1)
    angles = pos * freqs[None, :]                   # (n_pos, half)
    emb = jnp.concatenate([jnp.sin(angles), jnp.cos(angles)], axis=-1)  # (n_pos, 2*half)
    if d % 2 == 1:
        # pad one dim if odd
        emb = jnp.pad(emb, ((0, 0), (0, 1)))
    return emb.astype(dtype)  # (n_pos, d)


# ---------------------------
# Modules
# ---------------------------
class GRUGating(nn.Module):
    """GRU-style gating used in GTrXL to replace vanilla residuals."""
    dtype: Any = jnp.float32
    use_bias: bool = True

    @nn.compact
    def __call__(self, x, y):
        # x, y: (B, C)
        C = x.shape[-1]
        # Update gate z
        z = nn.sigmoid(
            nn.Dense(C, dtype=self.dtype, use_bias=self.use_bias, name="z_x")(x) +
            nn.Dense(C, dtype=self.dtype, use_bias=self.use_bias, name="z_y")(y)
        )
        # Reset gate r
        r = nn.sigmoid(
            nn.Dense(C, dtype=self.dtype, use_bias=self.use_bias, name="r_x")(x) +
            nn.Dense(C, dtype=self.dtype, use_bias=self.use_bias, name="r_y")(y)
        )
        # Candidate h~
        h_tilde = jnp.tanh(
            nn.Dense(C, dtype=self.dtype, use_bias=self.use_bias, name="h_x")(x * r) +
            nn.Dense(C, dtype=self.dtype, use_bias=self.use_bias, name="h_y")(y)
        )
        return (1.0 - z) * x + z * h_tilde


class RelPosSelfAttention(nn.Module):
    """Single-step relative-position (TXL) self-attention."""
    num_heads: int
    dtype: Any = jnp.float32
    dropout_rate: float = 0.1
    use_proj_bias: bool = True

    @nn.compact
    def __call__(self, x, mem, deterministic: Optional[bool]):
        """
        x:   (B, C) current step input (after pre-LN)
        mem: (B, M, C) layer memory of hidden states
        Returns: (B, C)
        """
        B, C = x.shape
        H = self.num_heads
        assert C % H == 0, f"hidden size {C} must be divisible by num_heads {H}"
        D = C // H

        # Concatenate memory with current step along time axis.
        # Keys/values are computed from [mem; x_now].
        if mem is None:
            L = 1
            cat = x[:, None, :]  # (B, 1, C)
        else:
            L = mem.shape[1] + 1
            cat = jnp.concatenate([mem, x[:, None, :]], axis=1)  # (B, L, C)

        # Linear projections
        q = nn.Dense(C, use_bias=self.use_proj_bias, dtype=self.dtype, name="q_proj")(x)          # (B, C)
        k = nn.Dense(C, use_bias=self.use_proj_bias, dtype=self.dtype, name="k_proj")(cat)        # (B, L, C)
        v = nn.Dense(C, use_bias=self.use_proj_bias, dtype=self.dtype, name="v_proj")(cat)        # (B, L, C)

        # Reshape to heads
        q = q.reshape(B, H, D)                     # (B, H, D)
        k = k.reshape(B, L, H, D)                  # (B, L, H, D)
        v = v.reshape(B, L, H, D)                  # (B, L, H, D)

        # TXL per-head global biases
        u_bias = self.param("u_bias", nn.initializers.zeros, (H, D), self.dtype)  # content
        v_bias = self.param("v_bias", nn.initializers.zeros, (H, D), self.dtype)  # positional

        # Relative pos encs for distances 0..L-1 (0=current, L-1=farthest)
        # We arrange them reversed so index j (0..L-1; mem->current) maps to r_{delta=L-1-j}.
        r = _sinusoidal_positions(L, D, dtype=self.dtype)            # (L, D)
        r = r[::-1, :]                                               # reverse to align keys oldest->current

        # Attention logits: A = (q+u)·k^T  +  (q+v)·r^T
        scale = 1.0 / jnp.sqrt(D).astype(self.dtype)

        # (B,H,D) x (B,L,H,D) -> (B,H,L)
        ac = jnp.einsum("bhd,blhd->bhl", q + u_bias[None, :, :], k)

        # (B,H,D) x (L,D) -> (B,H,L)
        bd = jnp.einsum("bhd,ld->bhl", q + v_bias[None, :, :], r)

        attn_logits = (ac + bd) * scale
        attn = jax.nn.softmax(attn_logits, axis=-1).astype(self.dtype)
        attn = nn.Dropout(self.dropout_rate)(attn, deterministic=deterministic)

        # Weighted sum of values: (B,H,L) x (B,L,H,D) -> (B,H,D)
        out = jnp.einsum("bhl,blhd->bhd", attn, v)
        out = out.reshape(B, C)

        out = nn.Dense(C, use_bias=self.use_proj_bias, dtype=self.dtype, name="o_proj")(out)
        out = nn.Dropout(rate=self.dropout_rate)(out, deterministic=deterministic)
        return out


class GTrXLMLP(nn.Module):
    config: GTrXLConfig

    @nn.compact
    def __call__(self, x, deterministic=None):
        C = x.shape[-1]
        x = nn.Dense(4 * C, dtype=self.config.dtype, use_bias=self.config.use_bias, name="c_fc")(x)
        x = nn.gelu(x, approximate=True)
        x = nn.Dense(C, dtype=self.config.dtype, use_bias=self.config.use_bias, name="c_proj")(x)
        x = nn.Dropout(self.config.dropout_rate)(x, deterministic)
        return x


class GTrXLBlock(nn.Module):
    config: GTrXLConfig

    def setup(self):
        self.ln_1 = nn.LayerNorm(
            epsilon=1e-5, dtype=self.config.dtype, use_bias=self.config.use_bias, name="ln_1"
        )
        self.attn = RelPosSelfAttention(
            self.config.num_heads,
            self.config.dtype,
            dropout_rate=self.config.dropout_rate,
        )
        self.gate_attn = GRUGating(self.config.dtype, self.config.use_bias)

        self.ln_2 = nn.LayerNorm(
            epsilon=1e-5, dtype=self.config.dtype, use_bias=self.config.use_bias, name="ln_2"
        )
        self.mlp = GTrXLMLP(self.config)
        self.gate_mlp = GRUGating(self.config.dtype, self.config.use_bias)

    def __call__(self, x, mem, deterministic):
        """
        x:   (B, C) single-step input
        mem: (B, M, C) layer memory (can be None or M=0)
        """
        h = self.ln_1(x)
        a = self.attn(h, mem, deterministic)
        x = self.gate_attn(x, a)

        h2 = self.ln_2(x)
        m = self.mlp(h2, deterministic)
        x = self.gate_mlp(x, m)
        return x


# ---------------------------
# RNN Cell
# ---------------------------
@flax.struct.dataclass
class GTrXLRNNCellCarry:
    mems: jnp.ndarray  # (batch, num_layers, mem_len, num_embeds)
    step: jnp.ndarray  # (batch,) or scalar if unbatched


class GTrXLRNNCell(nn.recurrent.RNNCellBase):
    """A recurrent Gated Transformer-XL cell compatible with RNNCellBase."""

    config: GTrXLConfig
    kernel_init: nn.initializers.Initializer = nn.initializers.lecun_normal()
    bias_init: nn.initializers.Initializer = nn.initializers.zeros_init()

    def initialize_carry(self, rng, input_shape):
        # input_shape: (..., features)
        batch_shape = input_shape[:-1]
        L = self.config.num_layers
        M = self.config.mem_len
        C = self.config.num_embeds

        if M > 0:
            mems = jnp.zeros(batch_shape + (L, M, C), dtype=self.config.dtype or jnp.float32)
        else:
            # Represent "no memory" with a zero-length mem axis for shape consistency
            mems = jnp.zeros(batch_shape + (L, 0, C), dtype=self.config.dtype or jnp.float32)

        step = jnp.zeros(batch_shape, dtype=jnp.int32)
        return GTrXLRNNCellCarry(mems=mems, step=step)

    @nn.compact
    def __call__(self, carry, inputs, deterministic: bool = True):
        """
        carry.mems: (B, L, M, C)
        inputs:     (B, C)
        """
        mems, step = carry.mems, carry.step
        B, C = inputs.shape
        L = self.config.num_layers
        M = self.config.mem_len

        x = inputs
        new_mems = mems

        for i in range(L):
            mem_i = mems[:, i] if M > 0 else None  # (B, M, C) or None
            x = GTrXLBlock(self.config, name=f"block_{i}")(x, mem_i, deterministic)

            # Update memory of layer i: append x, keep last M
            if M > 0:
                if mem_i is None or mem_i.shape[1] == 0:
                    updated = x[:, None, :]                                  # (B,1,C)
                else:
                    updated = jnp.concatenate([mem_i, x[:, None, :]], axis=1)  # (B,M+1,C)
                if updated.shape[1] > M:
                    updated = updated[:, -M:, :]  # keep the most recent M
                new_mems = new_mems.at[:, i].set(updated)

        # Optional final LN (mirrors your GPT cell’s ln_f)
        x = nn.LayerNorm(
            epsilon=1e-5, dtype=self.config.dtype, use_bias=self.config.use_bias, name="ln_f"
        )(x)

        new_carry = GTrXLRNNCellCarry(mems=new_mems, step=step + 1)
        return new_carry, x  # output: (B, C)

    @property
    def num_feature_axes(self):
        return 1

