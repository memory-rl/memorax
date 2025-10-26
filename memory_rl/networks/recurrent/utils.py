from typing import Callable, Tuple, Literal
from flax.typing import Dtype, Initializer
import jax
import jax.numpy as jnp
from jax.nn.initializers import lecun_normal
from jax.lib import xla_bridge

from flax import linen as nn
from flax.linen.dtypes import promote_dtype

from memory_rl.utils.typing import Array

Implementation = Literal["xla", "cudnn"]


def get_attention_implementation() -> Implementation:
    backend = xla_bridge.get_backend()
    platform = getattr(backend, "platform", "")
    if platform == "gpu":
        version = getattr(backend, "platform_version", "")
        if "cuda" in version.lower():
            return "cudnn"

        # Fallback detection using device kinds for NVIDIA GPUs.
        try:
            if any(
                "nvidia" in device.device_kind.lower() for device in jax.local_devices()
            ):
                return "cudnn"
        except Exception:  # pragma: no cover - best effort hardware detection
            pass

    return "xla"


def add_time_axis(x: jax.Array):
    return x[:, None, ...]


def remove_time_axis(x: jax.Array):
    return x.squeeze(1)


def get_time_axis_and_input_shape(inputs: jax.Array, num_feature_axes=1):
    time_axis = inputs.ndim - (num_feature_axes + 1)
    if time_axis < 0:
        time_axis += inputs.ndim
    input_shape = inputs.shape[:time_axis] + inputs.shape[time_axis + 1 :]
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


def uniform(minval, maxval):
    def init(key, shape, dtype):
        return jax.random.uniform(key, shape, minval=minval, maxval=maxval, dtype=dtype)

    return init


def f_bias_init(key, shape, dtype):
    """Initializes a weight matrix with a power law distribution."""
    num_heads, *_ = shape
    return jnp.linspace(3.0, 6.0, num_heads, dtype=dtype)


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
    block_size: int = 4
    use_bias: bool = True
    kernel_init: Initializer | None = None
    bias_init: Initializer = nn.initializers.zeros_init()
    dtype: Dtype | None = None
    param_dtype: Dtype = jnp.float32

    @nn.compact
    def __call__(self, x: Array) -> jax.Array:
        *batch, features = x.shape
        num_heads = features // self.block_size

        if features % self.block_size != 0:
            raise ValueError(
                f"head_dim ({features}) must be divisible by block_size ({self.block_size})."
            )

        kernel_init = self.kernel_init or small_init(self.features // num_heads)
        kernel = self.param(
            "kernel",
            kernel_init,
            (num_heads, self.block_size, self.block_size),
            self.param_dtype,
        )
        x, kernel = promote_dtype(x, kernel, dtype=self.dtype)
        x = x.reshape(*batch, num_heads, self.block_size)
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


def MultiHeadLayerNorm(
    use_scale: bool = True,
    use_bias: bool = False,
    eps: float = 1e-5,
    dtype: jnp.dtype = jnp.float32,
    axis: int = 1,
    **kwargs,
):
    return nn.vmap(
        nn.LayerNorm,
        variable_axes={"params": 0},
        in_axes=axis,
        out_axes=axis,
        split_rngs={"params": True},
    )(epsilon=eps, use_bias=use_bias, use_scale=use_scale, dtype=dtype, **kwargs)


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
