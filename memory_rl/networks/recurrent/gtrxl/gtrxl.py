from functools import partial  # pylint: disable=g-importing-member
from typing import (
    Any,
    TypeVar,
)

import jax
from jax import numpy as jnp
from jax import random

from flax.linen import initializers, LayerNorm, Dropout
from flax.linen.activation import sigmoid, tanh, softmax
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


def _split_heads(x: Array, n_heads: int) -> Array:
    b, t, d = x.shape
    d_head = d // n_heads
    x = x.reshape(b, t, n_heads, d_head)
    return jnp.transpose(x, (0, 2, 1, 3))  # (b, h, t, d_head)


def _merge_heads(x: Array) -> Array:
    b, h, t, d_head = x.shape
    return jnp.transpose(x, (0, 2, 1, 3)).reshape(b, t, h * d_head)


def _rel_shift(x: Array) -> Array:
    # x: (b, h, q_len, k_len) where k_len == r_len
    b, h, q_len, k_len = x.shape
    x = jnp.pad(x, ((0, 0), (0, 0), (0, 0), (1, 0)))  # (b, h, q, k+1)
    x = x.reshape(b, h, k_len + 1, q_len)
    x = x[:, :, 1:, :]
    x = x.reshape(b, h, q_len, k_len)
    return x


def sinusoidal_pos_emb(pos_seq: Array, dim: int) -> Array:
    # pos_seq: (L,) non-negative positions (e.g., distances), largest first.
    inv_freq = 1.0 / (10000 ** (jnp.arange(0, dim, 2, dtype=jnp.float32) / dim))
    sinusoid = jnp.einsum(
        "l,d->ld", pos_seq.astype(jnp.float32), inv_freq
    )  # (L, dim/2)
    pos_emb = jnp.concatenate(
        [jnp.sin(sinusoid), jnp.cos(sinusoid)], axis=-1
    )  # (L, dim)
    return pos_emb


class GRUGating(Module):
    """GRU-style gating used in GTrXL to replace residual connections."""

    features: int
    kernel_init: Initializer = default_kernel_init
    recurrent_kernel_init: Initializer = initializers.orthogonal()
    bias_init: Initializer = initializers.zeros_init()
    dtype: Dtype | None = None
    param_dtype: Dtype = jnp.float32

    @compact
    def __call__(self, x: Array, y: Array) -> Array:
        # x: residual stream (b, d); y: sublayer output (b, d)
        dense_y = partial(
            Dense,
            features=self.features,
            use_bias=True,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )
        dense_x = partial(
            Dense,
            features=self.features,
            use_bias=False,
            kernel_init=self.recurrent_kernel_init,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )
        z = sigmoid(dense_y(name="z_y")(y) + dense_x(name="z_x")(x))
        r = sigmoid(dense_y(name="r_y")(y) + dense_x(name="r_x")(x))
        h_tilde = tanh(dense_y(name="h_y")(y) + dense_x(name="h_x")(r * x))
        return (1.0 - z) * x + z * h_tilde


class RelativeMultiHeadAttention(Module):
    """Transformer-XL relative MHA (single-step query with memory)."""

    d_model: int
    n_heads: int
    d_head: int | None = None
    attn_dropout_rate: float = 0.0
    dtype: Dtype | None = None
    param_dtype: Dtype = jnp.float32
    kernel_init: Initializer = default_kernel_init
    bias_init: Initializer = initializers.zeros_init()

    @compact
    def __call__(self, h: Array, mem: Array, r: Array, *, deterministic: bool) -> Array:
        # h: (b, 1, d_model) query
        # mem: (b, mem_len, d_model) keys/values memory
        # r: (k_len, d_model) relative positional enc (k_len = mem_len + 1)
        d_model = self.d_model
        n_heads = self.n_heads
        d_head = self.d_head or (d_model // n_heads)

        assert d_model == n_heads * d_head, "d_model must equal n_heads * d_head"

        k_len = mem.shape[1] + h.shape[1]  # mem_len + 1
        assert r.shape[0] == k_len, "relative pos length must equal key length"

        proj_q = Dense(
            n_heads * d_head,
            use_bias=False,
            kernel_init=self.kernel_init,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="q",
        )
        proj_k = Dense(
            n_heads * d_head,
            use_bias=False,
            kernel_init=self.kernel_init,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="k",
        )
        proj_v = Dense(
            n_heads * d_head,
            use_bias=False,
            kernel_init=self.kernel_init,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="v",
        )
        proj_r = Dense(
            n_heads * d_head,
            use_bias=False,
            kernel_init=self.kernel_init,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="r",
        )
        proj_o = Dense(
            d_model,
            use_bias=False,
            kernel_init=self.kernel_init,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="o",
        )

        # Content and position biases (u, v) per head.
        r_w_bias = self.param(
            "r_w_bias", self.bias_init, (n_heads, d_head), self.param_dtype
        )
        r_r_bias = self.param(
            "r_r_bias", self.bias_init, (n_heads, d_head), self.param_dtype
        )

        # Build key/value from [mem, h]
        cat = jnp.concatenate([mem, h], axis=1)  # (b, k_len, d)

        q = _split_heads(proj_q(h), n_heads)  # (b, h, 1, d_head)
        k = _split_heads(proj_k(cat), n_heads)  # (b, h, k_len, d_head)
        v = _split_heads(proj_v(cat), n_heads)  # (b, h, k_len, d_head)
        r_head = proj_r(r).reshape(k_len, n_heads, d_head)  # (k_len, h, d_head)
        r_head = jnp.transpose(r_head, (1, 0, 2))  # (h, k_len, d_head)

        # Content-based term: (q + u) @ k^T
        qw = q + r_w_bias[None, :, None, :]  # (b, h, 1, d_head)
        ac = jnp.einsum("bhqd,bhkd->bhqk", qw, k)  # (b, h, 1, k_len)

        # Position-based term: (q + v) @ r^T then relative shift
        qr = q + r_r_bias[None, :, None, :]
        bd = jnp.einsum("bhqd,hkd->bhqk", qr, r_head)  # (b, h, 1, k_len)
        bd = _rel_shift(bd)

        attn_score = (ac + bd) / jnp.sqrt(
            jnp.array(d_head, dtype=self.dtype or jnp.float32)
        )
        attn_prob = softmax(attn_score, axis=-1)
        if self.attn_dropout_rate > 0:
            attn_prob = Dropout(
                rate=self.attn_dropout_rate,
                deterministic=deterministic,
                name="attn_dropout",
            )(attn_prob)

        attn_vec = jnp.einsum("bhqk,bhkd->bhqd", attn_prob, v)  # (b, h, 1, d_head)
        attn_vec = _merge_heads(attn_vec)  # (b, 1, d_model)
        out = proj_o(attn_vec)  # (b, 1, d_model)
        return out


class PositionwiseFF(Module):
    d_model: int
    d_inner: int
    dropout_rate: float = 0.0
    dtype: Dtype | None = None
    param_dtype: Dtype = jnp.float32
    kernel_init: Initializer = default_kernel_init
    bias_init: Initializer = initializers.zeros_init()

    @compact
    def __call__(self, x: Array, *, deterministic: bool) -> Array:
        x = Dense(
            self.d_inner,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="ff1",
        )(x)
        x = jax.nn.relu(x)
        if self.dropout_rate > 0:
            x = Dropout(
                rate=self.dropout_rate, deterministic=deterministic, name="ff_dropout"
            )(x)
        x = Dense(
            self.d_model,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="ff2",
        )(x)
        return x


class GTrXLBlock(Module):
    d_model: int
    n_heads: int
    d_head: int | None
    d_inner: int
    attn_dropout_rate: float
    dropout_rate: float
    dtype: Dtype | None
    param_dtype: Dtype
    kernel_init: Initializer
    bias_init: Initializer

    @compact
    def __call__(
        self, x: Array, mem: Array, r: Array, *, deterministic: bool
    ) -> tuple[Array, Array]:
        # Pre-LN -> Rel-MHA -> GRU-gated residual
        y = LayerNorm(dtype=self.dtype, name="ln_attn")(x)
        y = RelativeMultiHeadAttention(
            d_model=self.d_model,
            n_heads=self.n_heads,
            d_head=self.d_head,
            attn_dropout_rate=self.attn_dropout_rate,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
            name="attn",
        )(y, mem, r, deterministic=deterministic)
        if self.dropout_rate > 0:
            y = Dropout(
                rate=self.dropout_rate,
                deterministic=deterministic,
                name="attn_out_drop",
            )(y)
        x = GRUGating(
            self.d_model,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
            name="attn_gate",
        )(x.squeeze(axis=1), y.squeeze(axis=1))
        x = x[:, None, :]

        # Pre-LN -> FF -> GRU-gated residual
        y2 = LayerNorm(dtype=self.dtype, name="ln_ff")(x)
        y2 = PositionwiseFF(
            self.d_model,
            self.d_inner,
            self.dropout_rate,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
            name="ff",
        )(y2, deterministic=deterministic)
        if self.dropout_rate > 0:
            y2 = Dropout(
                rate=self.dropout_rate, deterministic=deterministic, name="ff_out_drop"
            )(y2)
        x = GRUGating(
            self.d_model,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
            name="ff_gate",
        )(x.squeeze(axis=1), y2.squeeze(axis=1))
        x = x[:, None, :]
        return x, mem  # mem returned unmodified; updated outside with new state


class GTrXLCell(RNNCellBase):
    """Gated Transformer-XL cell (single-token step with segment-level recurrence).

    Carry:
      A tuple of per-layer memories: ((b, mem_len, d_model), ... ) of length n_layers.

    Inputs:
      (b, d_model)

    Output:
      (b, d_model)
    """

    features: int  # == d_model
    n_layers: int
    n_heads: int
    d_head: int | None = None
    d_inner: int | None = None
    mem_len: int = 128
    dropout_rate: float = 0.0
    attn_dropout_rate: float = 0.0
    dtype: Dtype | None = None
    param_dtype: Dtype = jnp.float32
    kernel_init: Initializer = default_kernel_init
    bias_init: Initializer = initializers.zeros_init()
    carry_init: Initializer = initializers.zeros_init()

    @property
    def num_feature_axes(self) -> int:
        return 1

    def _build_pos_emb(self, mem_len: int, q_len: int, dim: int) -> Array:
        k_len = mem_len + q_len
        pos_seq = jnp.arange(k_len - 1, -1, -1)  # distances: [k_len-1 ... 0]
        return sinusoidal_pos_emb(pos_seq, dim)  # (k_len, dim)

    @compact
    def __call__(
        self, carry: tuple[Array, ...], inputs: Array
    ) -> tuple[tuple[Array, ...], Array]:
        # inputs: (b, d_model)
        assert inputs.shape[-1] == self.features, "inputs last dim must equal d_model"

        b = inputs.shape[0]
        d_model = self.features
        d_inner = self.d_inner or (4 * d_model)
        d_head = self.d_head or (d_model // self.n_heads)
        assert d_model == self.n_heads * d_head, "d_model must equal n_heads * d_head"

        mems = list(carry)  # per-layer: (b, mem_len, d_model)
        h = inputs[:, None, :]  # (b, 1, d_model)
        r = self._build_pos_emb(self.mem_len, 1, d_model)  # (mem_len+1, d_model)

        deterministic = True  # Dropout disabled by default inside a scan unless user opts in via attributes + RNG.

        new_mems = []

        for layer_idx in range(self.n_layers):
            block = GTrXLBlock(
                d_model=d_model,
                n_heads=self.n_heads,
                d_head=d_head,
                d_inner=d_inner,
                attn_dropout_rate=self.attn_dropout_rate,
                dropout_rate=self.dropout_rate,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                kernel_init=self.kernel_init,
                bias_init=self.bias_init,
                name=f"layer_{layer_idx}",
            )
            h, _ = block(
                h, mems[layer_idx], r, deterministic=deterministic
            )  # h: (b, 1, d_model)

            # Update memory with the *post-block* hidden state (Transformer-XL stores layer outputs).
            if self.mem_len > 0:
                cur = h  # (b, 1, d_model)
                mem_cat = jnp.concatenate(
                    [mems[layer_idx], cur], axis=1
                )  # (b, mem_len+1, d_model)
                new_mem = mem_cat[:, -self.mem_len :, :]  # keep most recent mem_len
            else:
                # Keep an empty memory of correct shape.
                new_mem = jnp.zeros((b, 0, d_model), dtype=self.param_dtype)
            new_mems.append(new_mem)

        y = h.squeeze(axis=1)  # (b, d_model)
        return tuple(new_mems), y

    @nowrap
    def initialize_carry(
        self, rng: PRNGKey, input_shape: tuple[int, ...]
    ) -> tuple[Array, ...]:
        # input_shape: (*batch, d_model)
        batch_dims = input_shape[:-1]
        mem_shape = batch_dims + (self.mem_len, self.features)
        mems = tuple(
            self.carry_init(random.fold_in(rng, i), mem_shape, self.param_dtype)
            for i in range(self.n_layers)
        )
        return mems
