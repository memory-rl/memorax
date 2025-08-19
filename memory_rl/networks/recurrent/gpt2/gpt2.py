from dataclasses import dataclass
from typing import Any, Optional

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import jax.lax as lax


@dataclass(frozen=True)
class GPTConfig:
    block_size: int
    vocab_size: int
    num_layers: int
    num_heads: int
    num_embeds: int
    dropout_rate: float = 0.1
    use_bias: bool = True
    dtype: Optional[str] = None


class SelfAttention(nn.Module):

    num_heads: int
    dtype: Any = jnp.float32
    dropout_rate: float = 0.1
    deterministic: Optional[bool] = None
    use_proj_bias: bool = True

    @nn.compact
    def __call__(self, x, deterministic, prev_k, prev_v, pos):
        # x: (B, C) single step input
        # prev_k, prev_v: (B, max_seq, num_heads, head_dim)
        C = x.shape[-1]

        assert C % self.num_heads == 0, f"Input dimension {C} is not divisible by num_heads {self.num_heads}"
        head_dim = C // self.num_heads

        qkv = nn.Dense(3 * C, use_bias=self.use_proj_bias, dtype=self.dtype, name="c_attn")(x)  # (B, 3*C)
        qkv = qkv.reshape(x.shape[0], 3, self.num_heads, head_dim)  # (B, 3, num_heads, head_dim)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]  # each (B, num_heads, head_dim)
        # Expand to (B, 1, num_heads, head_dim) for time axis
        k = k[:, None, :, :]
        v = v[:, None, :, :]
        # always use the full cache, but mask out future positions
        max_seq = prev_k.shape[1]
        # Build mask: (B, 1, max_seq)
        mask = jnp.arange(max_seq)[None, None, :] <= pos[..., None, None]
        # query is current, keys/values are all cached
        scale = 1.0 / jnp.sqrt(head_dim).astype(self.dtype)
        attn = jnp.einsum("bhd,bthd->bht", q, prev_k) * scale
        attn = jnp.where(mask, attn, jnp.finfo(self.dtype).min)
        attn = jax.nn.softmax(attn, axis=-1).astype(self.dtype)
        attn = nn.Dropout(self.dropout_rate)(attn, deterministic=deterministic)
        out = jnp.einsum("bht,bthd->bhd", attn, prev_v)
        out = out.reshape(x.shape[0], C)
        out = nn.Dense(C, use_bias=self.use_proj_bias, dtype=self.dtype, name="c_proj")(out)
        out = nn.Dropout(rate=self.dropout_rate)(out, deterministic=deterministic)
        return out, k, v


class MLP(nn.Module):
    config: GPTConfig

    @nn.compact
    def __call__(self, x, deterministic=None):
        C = x.shape[-1]
        x = nn.Dense(
            4 * C, dtype=self.config.dtype, use_bias=self.config.use_bias, name="c_fc"
        )(x)
        x = nn.gelu(x, approximate=True)
        x = nn.Dense(
            C, dtype=self.config.dtype, use_bias=self.config.use_bias, name="c_proj"
        )(x)
        x = nn.Dropout(self.config.dropout_rate)(x, deterministic)
        return x


class Block(nn.Module):
    config: GPTConfig

    def setup(self):
        self.ln_1 = nn.LayerNorm(
            epsilon=1e-5, dtype=self.config.dtype, use_bias=self.config.use_bias
        )
        self.attn = SelfAttention(
            self.config.num_heads,
            self.config.dtype,
            dropout_rate=self.config.dropout_rate,
        )
        self.ln_2 = nn.LayerNorm(
            epsilon=1e-5, dtype=self.config.dtype, use_bias=self.config.use_bias
        )
        self.mlp = MLP(self.config)

    def __call__(self, x, deterministic, prev_k, prev_v, pos):
        # x: (B, C) single step input
        # prev_k, prev_v: (B, max_seq, num_heads, head_dim)
        h = self.ln_1(x)
        attn_out, k, v = self.attn(h, deterministic, prev_k, prev_v, pos)
        x = x + attn_out
        x = x + self.mlp(self.ln_2(x), deterministic)
        return x, k, v


# class GPT(nn.Module):
#     config: GPTConfig

#     @nn.compact
#     def __call__(self, idx, deterministic=None):
#         B, T = idx.shape
#         assert (
#             T <= self.config.block_size
#         ), f"Cannot forward sequence of length {T}, block size is only {self.block_size}"

#         pos = jnp.arange(0, T)[None]
#         attn_mask = nn.make_causal_mask(idx, dtype=bool)

#         wte = nn.Embed(
#             self.config.vocab_size,
#             self.config.num_embeds,
#             dtype=self.config.dtype,
#             name="wte",
#         )
#         wpe = nn.Embed(
#             self.config.block_size,
#             self.config.num_embeds,
#             dtype=self.config.dtype,
#             name="wpe",
#         )

#         token_embed = wte(idx)  # [B, T, num_embeds]
#         pos_embed = wpe(pos)  # [1, T, num_embeds]
#         x = nn.Dropout(self.config.dropout_rate)(token_embed + pos_embed, deterministic)

#         for i in range(self.config.num_layers):
#             x = Block(self.config, name=str(i))(
#                 x, attn_mask, deterministic=deterministic
#             )

#         x = nn.LayerNorm(
#             1e-5, dtype=self.config.dtype, use_bias=self.config.use_bias, name="ln_f"
#         )(x)
#         logits = wte.attend(x)
#         return logits

#     def init(self, rng):
#         """
#         by jitting init, traced values instead of concrete values are used
#         which saves memory (since un-jitted model may not fit in memory)
#         """
#         tokens = jnp.zeros((2, self.config.block_size), dtype=jnp.uint16)
#         params = jax.jit(super().init, static_argnums=(2,))(rng, tokens, True)
#         return params


@flax.struct.dataclass
class GPTRNNCellCarry:
    kv_cache_k: jnp.ndarray  # (batch, num_blocks, max_seq, num_heads, head_dim)
    kv_cache_v: jnp.ndarray  # (batch, num_blocks, max_seq, num_heads, head_dim)
    pos: jnp.ndarray  # (batch,) or () if not batched


class GPTRNNCell(nn.recurrent.RNNCellBase):
    """A recurrent GPT2 cell compatible with RNNCellBase."""

    max_sequence_length: int
    config: GPTConfig

    def initialize_carry(self, rng, input_shape):
        # input_shape: (..., features)
        batch_shape = input_shape[:-1]
        num_blocks = self.config.num_layers
        num_heads = self.config.num_heads
        head_dim = self.config.num_embeds // self.config.num_heads
        max_seq = self.max_sequence_length
        # Preallocate full cache for all blocks: (num_blocks, batch, max_seq, num_heads, head_dim)
        kv_cache_k = jnp.zeros(batch_shape + (num_blocks, max_seq, num_heads, head_dim), dtype=self.config.dtype or jnp.float32)
        kv_cache_v = jnp.zeros(batch_shape + (num_blocks, max_seq, num_heads, head_dim), dtype=self.config.dtype or jnp.float32)
        pos = jnp.zeros(batch_shape, dtype=jnp.int32)
        return GPTRNNCellCarry(kv_cache_k=kv_cache_k, kv_cache_v=kv_cache_v, pos=pos)

    @nn.compact
    def __call__(self, carry, inputs, deterministic=True):

        kv_cache_k, kv_cache_v, pos = carry.kv_cache_k, carry.kv_cache_v, carry.pos
        # inputs: (batch, features)
        # Add positional embedding for current position
        pos_emb = nn.Embed(
            self.max_sequence_length,
            self.config.num_embeds,
            dtype=self.config.dtype,
            name="wpe",
        )(pos[..., None])  # (batch, 1, num_embeds)
        x = inputs + pos_emb.squeeze(axis=-2)  # (batch, num_embeds)
        x = nn.Dropout(self.config.dropout_rate)(x, deterministic=deterministic)

        new_kv_cache_k = kv_cache_k
        new_kv_cache_v = kv_cache_v
        num_blocks = self.config.num_layers
        for i in range(num_blocks):
            prev_k_i = kv_cache_k[:, i]
            prev_v_i = kv_cache_v[:, i]
            # cur_pos: (batch,) or scalar
            cur_pos = pos if pos.shape == () else pos[0]

            x_out, k_new, v_new = Block(self.config, name=f"block_{i}")(x, deterministic, prev_k_i, prev_v_i, cur_pos)
            k_new_last = k_new[..., -1:, :, :]
            v_new_last = v_new[..., -1:, :, :]
            k_update = lax.dynamic_update_slice(prev_k_i, k_new_last, (0, cur_pos, 0, 0))
            v_update = lax.dynamic_update_slice(prev_v_i, v_new_last, (0, cur_pos, 0, 0))
            new_kv_cache_k = new_kv_cache_k.at[:, i].set(k_update)
            new_kv_cache_v = new_kv_cache_v.at[:, i].set(v_update)
            x = x_out
        x = nn.LayerNorm(
            epsilon=1e-5,
            dtype=self.config.dtype,
            use_bias=self.config.use_bias,
            name="ln_f",
        )(x)
        # Output is the current position's output
        output = x  # (batch, num_embeds)
        new_carry = GPTRNNCellCarry(
            kv_cache_k=new_kv_cache_k,
            kv_cache_v=new_kv_cache_v,
            pos=pos + 1,
        )
        return new_carry, output

    @property
    def num_feature_axes(self):
        return 1  # For now only 1 is supported


if __name__ == "__main__":
    print("Testing GPT...")

    batch_size = 64
    seq_length = 50
    features = 256  # num_embeds
    cell = flax.linen.RNN(
        GPTRNNCell(
            config=GPTConfig(
                block_size=1,
                vocab_size=1,
                num_layers=4,
                num_heads=4,
                num_embeds=features,
                dropout_rate=0.1,
                use_bias=True
            ),
            max_sequence_length=seq_length,
        )
    )
    x = jnp.ones((batch_size, seq_length, features), dtype=jnp.float32)

    y, variables = cell.init_with_output(jax.random.PRNGKey(0), x)
    print("Output shape:", y.shape)
    # print("Output:", y)