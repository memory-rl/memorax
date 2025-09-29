from typing import Any
import jax.numpy as jnp
from flax import linen as nn
from flax.linen.attention import dot_product_attention
from flax.linen.recurrent import RNNCellBase

def gelu_new(x):
    return 0.5 * x * (1.0 + jnp.tanh(jnp.sqrt(2.0 / jnp.pi) * (x + 0.044715 * x**3)))

class GPT2Attention(nn.Module):
    features: int
    n_head: int
    max_length: int
    attn_pdrop: float
    resid_pdrop: float
    dtype: Any
    param_dtype: Any
    kernel_init: Any
    bias_init: Any

    @nn.compact
    def __call__(self, x, past_k, past_v, idx: jnp.ndarray, deterministic: bool):
        head_dim = self.features // self.n_head
        c_attn = nn.Dense(
            3 * self.features,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
            name="c_attn",
        )
        c_proj = nn.Dense(
            self.features,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
            name="c_proj",
        )
        qkv = c_attn(x)
        q, k, v = jnp.split(qkv, 3, axis=-1)
        batch_shape = x.shape[:-1]
        q = q.reshape(batch_shape + (1, self.n_head, head_dim))
        k_step = k.reshape(batch_shape + (self.n_head, head_dim))
        v_step = v.reshape(batch_shape + (self.n_head, head_dim))
        oh = nn.one_hot(idx, self.max_length, dtype=self.dtype)
        oh_kv = oh[..., :, None, None]
        k_all = past_k * (1.0 - oh_kv) + (oh_kv * k_step[..., None, :, :])
        v_all = past_v * (1.0 - oh_kv) + (oh_kv * v_step[..., None, :, :])
        steps = jnp.arange(self.max_length)
        mask = (steps <= idx[..., None])[..., None, None, :]
        rng = None if deterministic else self.make_rng("dropout")
        y = dot_product_attention(
            q, k_all, v_all,
            mask=mask,
            broadcast_dropout=True,
            dropout_rng=rng,
            dropout_rate=self.attn_pdrop,
            deterministic=deterministic,
            dtype=self.dtype,
        )
        y = y.reshape(batch_shape + (self.n_head * head_dim,))
        y = c_proj(y)
        y = nn.Dropout(rate=self.resid_pdrop)(y, deterministic=deterministic)
        return y, k_step, v_step

class GPT2MLP(nn.Module):
    features: int
    inner_dim: int
    resid_pdrop: float
    dtype: Any
    param_dtype: Any
    kernel_init: Any
    bias_init: Any

    @nn.compact
    def __call__(self, x, deterministic: bool):
        c_fc = nn.Dense(
            self.inner_dim,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
            name="c_fc",
        )
        c_proj = nn.Dense(
            self.features,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
            name="c_proj",
        )
        x = c_fc(x)
        x = gelu_new(x)
        x = c_proj(x)
        x = nn.Dropout(rate=self.resid_pdrop)(x, deterministic=deterministic)
        return x

class GPT2Block(nn.Module):
    features: int
    n_head: int
    n_inner: int
    max_length: int
    attn_pdrop: float
    resid_pdrop: float
    layer_norm_epsilon: float
    dtype: Any
    param_dtype: Any
    kernel_init: Any
    bias_init: Any

    @nn.compact
    def __call__(self, x, past_k, past_v, idx: jnp.ndarray, deterministic: bool):
        ln_1 = nn.LayerNorm(epsilon=self.layer_norm_epsilon, dtype=self.dtype, param_dtype=self.param_dtype, name="ln_1")
        ln_2 = nn.LayerNorm(epsilon=self.layer_norm_epsilon, dtype=self.dtype, param_dtype=self.param_dtype, name="ln_2")
        attn = GPT2Attention(
            features=self.features,
            n_head=self.n_head,
            max_length=self.max_length,
            attn_pdrop=self.attn_pdrop,
            resid_pdrop=self.resid_pdrop,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
            name="attn",
        )
        inner = self.n_inner if self.n_inner > 0 else 4 * self.features
        mlp = GPT2MLP(
            features=self.features,
            inner_dim=inner,
            resid_pdrop=self.resid_pdrop,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
            name="mlp",
        )
        h, k_new, v_new = attn(ln_1(x), past_k, past_v, idx, deterministic)
        x = x + h
        x = x + mlp(ln_2(x), deterministic)
        return x, k_new, v_new

class GPT2Cell(RNNCellBase):
    features: int
    n_layer: int = 12
    n_head: int = 12
    n_inner: int = 0
    max_length: int = 1024
    embd_pdrop: float = 0.1
    attn_pdrop: float = 0.1
    resid_pdrop: float = 0.1
    layer_norm_epsilon: float = 1e-5
    initializer_range: float = 0.02
    dtype: Any = jnp.float32
    param_dtype: Any = jnp.float32
    kernel_init: Any = nn.initializers.normal(stddev=0.02)
    bias_init: Any = nn.initializers.zeros_init()
    carry_init: Any = None

    @property
    def num_feature_axes(self) -> int:
        return 1

    def initialize_carry(self, rng, input_shape):
        batch_shape = input_shape[:-1]
        head_dim = self.features // self.n_head
        k = jnp.zeros(batch_shape + (self.n_layer, self.max_length, self.n_head, head_dim), dtype=self.dtype)
        v = jnp.zeros(batch_shape + (self.n_layer, self.max_length, self.n_head, head_dim), dtype=self.dtype)
        t = jnp.zeros(batch_shape, dtype=jnp.int32)
        return (k, v, t)

    @nn.compact
    def __call__(self, carry, inputs):
        k_cache, v_cache, t = carry
        x = inputs
        if x.shape[-1] != self.features:
            x = nn.Dense(
                self.features,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                kernel_init=self.kernel_init,
                bias_init=self.bias_init,
                name="in_proj",
            )(x)
        deterministic = not self.has_rng("dropout")
        x = nn.Dropout(rate=self.embd_pdrop, name="drop")(x, deterministic=deterministic)
        for i in range(self.n_layer):
            past_k = k_cache[..., i, :, :, :]
            past_v = v_cache[..., i, :, :, :]
            x, k_new, v_new = GPT2Block(
                features=self.features,
                n_head=self.n_head,
                n_inner=self.n_inner,
                max_length=self.max_length,
                attn_pdrop=self.attn_pdrop,
                resid_pdrop=self.resid_pdrop,
                layer_norm_epsilon=self.layer_norm_epsilon,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                kernel_init=nn.initializers.normal(stddev=self.initializer_range),
                bias_init=self.bias_init,
                name=f"h_{i}",
            )(x, past_k, past_v, t, deterministic)
            oh = nn.one_hot(t, self.max_length, dtype=self.dtype)
            oh_cache = oh[..., :, None, None]
            k_layer = k_cache[..., i, :, :, :]
            v_layer = v_cache[..., i, :, :, :]
            k_layer = k_layer * (1.0 - oh_cache) + (oh_cache * k_new[..., None, :, :])
            v_layer = v_layer * (1.0 - oh_cache) + (oh_cache * v_new[..., None, :, :])
            k_cache = k_cache.at[..., i, :, :, :].set(k_layer)
            v_cache = v_cache.at[..., i, :, :, :].set(v_layer)
        x = nn.LayerNorm(epsilon=self.layer_norm_epsilon, dtype=self.dtype, param_dtype=self.param_dtype, name="ln_f")(x)
        new_carry = (k_cache, v_cache, t + jnp.asarray(1, dtype=jnp.int32))
        return new_carry, x

