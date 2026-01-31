from typing import Callable, Literal, Tuple

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.linen.dtypes import promote_dtype
from flax.typing import Dtype, Initializer
from jax.nn.initializers import lecun_normal

from memorax.utils.typing import Array

Implementation = Literal["xla", "cudnn"]


def get_attention_implementation() -> tuple[Implementation, jnp.dtype]:
    backend = jax.default_backend()
    if backend == "gpu":

        try:
            if any(
                "nvidia" in device.device_kind.lower() for device in jax.local_devices()
            ):
                return "cudnn", jnp.bfloat16
        except Exception:
            pass

    return "xla", jnp.float32


def get_attention_mask(mask, initial_carry, memory_mask, context_length, num_heads):
    """Compute attention mask with position information for positional embeddings.

    Returns:
        Tuple of (combined_mask, query_input, key_input) where:
            - combined_mask: combined attention and causal mask (B, NH, T, S)
            - query_input: query positions (B, T)
            - key_input: key positions (B, M + context_length)
    """
    B, T = mask.shape
    _, M, *_ = memory_mask.shape

    query_mask = (
        jnp.cumsum(mask.astype(jnp.int32), axis=1)
        + jnp.max(
            jnp.cumsum(
                jnp.concatenate([memory_mask, initial_carry.mask], axis=1), axis=1
            ),
            axis=1,
        )[..., None]
    )

    key_mask = jnp.concatenate(
        [memory_mask, initial_carry.mask, mask], axis=1, dtype=jnp.int32
    )
    key_mask = jnp.cumsum(key_mask, axis=1)
    key_mask = key_mask[:, -(M + context_length) :]

    attention_mask = nn.make_attention_mask(query_mask, key_mask, pairwise_fn=jnp.equal)

    query_input = jnp.arange(T) + M + context_length
    query_input = jnp.broadcast_to(query_input, (B, T))
    key_input = jnp.arange(M + context_length + T)
    key_input = jnp.broadcast_to(key_input, (B, M + context_length + T))
    key_input = key_input[:, -(M + context_length) :]
    causal_mask = nn.make_attention_mask(
        query_input, key_input, pairwise_fn=jnp.greater_equal
    )

    B, _, T, S = attention_mask.shape
    attention_mask = jnp.broadcast_to(attention_mask, (B, num_heads, T, S))

    B, _, T, S = causal_mask.shape
    causal_mask = jnp.broadcast_to(causal_mask, (B, num_heads, T, S))

    combined_mask = nn.combine_masks(attention_mask, causal_mask, dtype=jnp.bool)
    return combined_mask, query_input, key_input


def add_time_axis(x: jax.Array):
    return x[:, None, ...]


def remove_time_axis(x: jax.Array):
    return x.squeeze(1)


def add_feature_axis(x: jax.Array):
    return x[..., None]


def remove_feature_axis(x: jax.Array):
    return x.squeeze(-1)


def get_time_axis(inputs: jax.Array, num_feature_axes=1):
    time_axis = inputs.ndim - (num_feature_axes + 1)
    if time_axis < 0:
        time_axis += inputs.ndim
    return time_axis


def get_input_shape(inputs: jax.Array, num_feature_axes=1):
    time_axis = get_time_axis(inputs, num_feature_axes)
    input_shape = inputs.shape[:time_axis] + inputs.shape[time_axis + 1 :]
    return input_shape


def get_time_axis_and_input_shape(inputs: jax.Array, num_feature_axes=1):
    time_axis = get_time_axis(inputs, num_feature_axes)
    input_shape = get_input_shape(inputs, num_feature_axes)
    return time_axis, input_shape


def broadcast_mask(mask: jax.Array, carry: jax.Array) -> jax.Array:
    while mask.ndim != carry.ndim:
        mask = mask[..., None] if mask.ndim < carry.ndim else mask[..., 0]
    return mask


def mask_carry(mask, carry, initial_carry):
    return jax.tree.map(
        lambda initial_carry, carry: jnp.where(
            broadcast_mask(mask, carry), initial_carry, carry
        ),
        initial_carry,
        carry,
    )


def kaiming_uniform():
    return nn.initializers.variance_scaling(2.0 / (1 + 5), "fan_in", "uniform")


def xavier_uniform():
    return nn.initializers.variance_scaling(1.0, "fan_avg", "uniform")


def uniform_init(min_val, max_val):
    def init(key, shape, dtype):
        return jax.random.uniform(
            key, shape, minval=min_val, maxval=max_val, dtype=dtype
        )

    return init


def powerlaw_init(num_heads, head_dim):
    """Initializes a weight matrix with a power law distribution."""

    def init(key, shape, dtype):
        x = jnp.arange(head_dim) / jnp.maximum(1.0, head_dim - 1)
        v = -(-5.0 + 12.0 * (x**0.3))
        b = jnp.tile(v, reps=(num_heads,))
        return b.astype(dtype)

    return init


def linspace_init(start, stop):
    def init(key, shape, dtype):
        num_dims, *_ = shape
        return jnp.linspace(start, stop, num_dims, dtype=dtype)

    return init


def small_init(dim):
    def init(key, shape, dtype):
        std = jnp.sqrt(2.0 / 5.0 / dim)
        return jax.random.normal(key, shape, dtype) * std

    return init


def wang_init(dim, num_blocks):
    def init(key, shape, dtype):
        std = 2.0 / (num_blocks * jnp.sqrt(dim))
        return jax.random.normal(key, shape, dtype) * std

    return init


class BlockDiagonalDense(nn.Module):
    features: int
    num_heads: int
    use_bias: bool = True
    kernel_init: Initializer | None = None
    bias_init: Initializer = nn.initializers.zeros_init()
    dtype: Dtype | None = None
    param_dtype: Dtype = jnp.float32

    @nn.compact
    def __call__(self, x: Array) -> jax.Array:
        *batch, features = x.shape
        block_size = features // self.num_heads

        kernel_init = self.kernel_init or small_init(block_size)
        kernel = self.param(
            "kernel",
            kernel_init,
            (self.num_heads, block_size, block_size),
            self.param_dtype,
        )
        x = x.reshape(*batch, self.num_heads, -1)
        x = jnp.einsum("...hd,hod->...ho", x, kernel)
        x = x.reshape(*batch, -1)

        if self.use_bias:
            bias = self.param(
                "bias",
                self.bias_init,
                (self.features,),
                self.param_dtype,
            )
            bias = jnp.broadcast_to(bias, x.shape)
            x = x + bias

        return x


class MultiHeadLayerNorm(nn.Module):
    eps: float = 1e-5
    use_scale: bool = True
    use_bias: bool = False
    residual_weight: bool = True
    dtype: Dtype | None = None
    param_dtype: Dtype = jnp.float32

    @nn.compact
    def __call__(self, x):
        B, NH, S, DH = x.shape

        y = nn.vmap(
            nn.LayerNorm,
            variable_axes={"params": 0},
            split_rngs={"params": True},
            in_axes=1,
            out_axes=1,
        )(
            epsilon=self.eps,
            use_scale=False,
            use_bias=False,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )(x)

        if self.use_scale:
            gamma = self.param(
                "weight", nn.initializers.zeros_init(), (NH, DH), self.param_dtype
            )
            scale = (1.0 + gamma) if self.residual_weight else gamma
            y = y * scale[None, :, None, :].astype(y.dtype)

        if self.use_bias:
            beta = self.param(
                "bias", nn.initializers.zeros_init(), (NH, DH), self.param_dtype
            )
            y = y + beta[None, :, None, :].astype(y.dtype)

        return y


class CausalConv1d(nn.Module):
    features: int
    kernel_size: int = 4
    use_bias: bool = True
    param_dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x: jnp.ndarray, state: jnp.ndarray) -> tuple:
        kernel = self.param(
            "kernel",
            kaiming_uniform(),
            (self.kernel_size, self.features),
            self.param_dtype,
        )

        conv_state = jnp.concatenate([state[:, 1:, :], x], axis=1)
        y = jnp.einsum("bkf,kf->bf", conv_state, kernel)[:, None, :]

        if self.use_bias:
            bias = self.param(
                "bias", nn.initializers.zeros_init(), (self.features,), self.param_dtype
            )
            y = y + bias
        return conv_state, y


class ParallelCausalConv1d(nn.Module):
    features: int
    kernel_size: int = 4
    use_bias: bool = True
    param_dtype: jnp.dtype = jnp.float32
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        *_, feature_group_count = x.shape
        padding = self.kernel_size - 1
        x = nn.Conv(
            features=self.features,
            kernel_size=self.kernel_size,
            kernel_init=kaiming_uniform(),
            bias_init=uniform_init(
                min_val=-1.0 / jnp.sqrt(self.kernel_size),
                max_val=1.0 / jnp.sqrt(self.kernel_size),
            ),
            feature_group_count=feature_group_count,
            padding=[(padding, 0)],
            use_bias=self.use_bias,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )(x)
        return x


def make_hippo(n: int) -> jnp.ndarray:
    p = jnp.sqrt(1.0 + 2.0 * jnp.arange(n))
    a = p[:, None] * p[None, :]
    a = jnp.tril(a) - jnp.diag(jnp.arange(n))
    return -a


def make_nplr_hippo(n: int) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    a = make_hippo(n)
    p = jnp.sqrt(jnp.arange(n) + 0.5)
    b = jnp.sqrt(2.0 * jnp.arange(n) + 1.0)
    return a, p, b


def make_dplr_hippo(
    n: int,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    a, p, b = make_nplr_hippo(n)
    s = a + p[:, None] * p[None, :]
    s_diag = jnp.diag(s)
    lambda_real = jnp.mean(s_diag) * jnp.ones_like(s_diag)
    lambda_imag, v = jnp.linalg.eigh(s * (-1j))
    p = v.conj().T @ p
    b_orig = b
    b = v.conj().T @ b
    return lambda_real + 1j * lambda_imag, p, v.conj().T @ b, v, b_orig


def log_step_initializer(dt_min: float = 0.001, dt_max: float = 0.1) -> Callable:
    def init(key, shape):
        return jax.random.uniform(key, shape) * (
            jnp.log(dt_max) - jnp.log(dt_min)
        ) + jnp.log(dt_min)

    return init


def init_log_steps(key, input_tuple):
    h, dt_min, dt_max = input_tuple
    logs = []
    for _ in range(h):
        key, sk = jax.random.split(key)
        logs.append(log_step_initializer(dt_min, dt_max)(sk, (1,)))
    return jnp.asarray(logs)


def init_v_inv_b(
    init_fun: Callable, rng: jax.Array, shape: Tuple[int, int], vinv: jnp.ndarray
) -> jnp.ndarray:
    b = init_fun(rng, shape)
    vinv_b = vinv.astype(jnp.complex64) @ b.astype(jnp.complex64)
    r = vinv_b.real
    i = vinv_b.imag
    return jnp.concatenate([r[..., None], i[..., None]], axis=-1)


def truncated_standard_normal(key, shape):
    h, p, _ = shape
    cs = []
    for _ in range(h):
        key, sk = jax.random.split(key)
        cs.append(lecun_normal()(sk, (1, p, 2)))
    return jnp.asarray(cs)[:, 0]


def init_cv(
    init_fun: Callable, rng: jax.Array, shape: Tuple[int, int, int], v: jnp.ndarray
) -> jnp.ndarray:
    c_ = init_fun(rng, shape)
    c = c_[..., 0] + 1j * c_[..., 1]
    cv = c @ v.astype(jnp.complex64)
    r = cv.real
    i = cv.imag
    return jnp.concatenate([r[..., None], i[..., None]], axis=-1)


def discretize_bilinear(
    lam: jnp.ndarray, b_tilde: jnp.ndarray, delta: jnp.ndarray
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    ident = jnp.ones(lam.shape[0], dtype=jnp.complex64)
    bl = 1.0 / (ident - (delta / 2.0) * lam)
    lambda_bar = bl * (ident + (delta / 2.0) * lam)
    b_bar = (bl * delta)[..., None] * b_tilde
    return lambda_bar, b_bar


def discretize_zoh(
    lam: jnp.ndarray, b_tilde: jnp.ndarray, delta: jnp.ndarray
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    ident = jnp.ones(lam.shape[0], dtype=jnp.complex64)
    lambda_bar = jnp.exp(lam * delta)
    b_bar = (1.0 / lam * (lambda_bar - ident))[..., None] * b_tilde
    return lambda_bar, b_bar
