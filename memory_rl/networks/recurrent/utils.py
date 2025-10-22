from typing import Callable, Tuple, Literal
from flax.typing import Dtype
import jax
import jax.numpy as jnp
from jax.nn.initializers import lecun_normal
from jax.lib import xla_bridge

from flax import linen as nn

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
    use_channel_mixing: bool = False
    dtype: Dtype | None = None

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        if self.use_channel_mixing:
            groups = 1
        else:
            groups = x.shape[-1]
        padding = self.kernel_size - 1
        fan_in = self.kernel_size * (x.shape[-1] // groups)
        x = nn.Conv(
            features=self.features,
            kernel_size=(self.kernel_size,),
            kernel_init=kaiming_uniform(),
            bias_init=uniform(
                minval=-1.0 / jnp.sqrt(fan_in), maxval=1.0 / jnp.sqrt(fan_in)
            ),
            feature_group_count=groups,
            padding=[(padding, 0)],
            use_bias=self.use_bias,
            dtype=self.dtype,
            name="causal_conv_1d",
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
