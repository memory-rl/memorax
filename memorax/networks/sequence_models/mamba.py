from functools import partial
from typing import Tuple

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.typing import Dtype, Initializer

from memorax.utils.typing import Array, Carry

from .memoroid import MemoroidCellBase


def _A_log_init(A_min=1.0, A_max=16.0):
    def init(key, shape):
        return jax.random.uniform(
            key, shape, minval=jnp.log(A_min), maxval=jnp.log(A_max)
        )

    return init


def _dt_bias_init(dt_min=0.001, dt_max=0.1):
    def init(key, shape):
        dt = jnp.exp(
            jax.random.uniform(key, shape)
            * (jnp.log(dt_max) - jnp.log(dt_min))
            + jnp.log(dt_min)
        )
        return dt + jnp.log(-jnp.expm1(-dt))

    return init


class MambaCell(MemoroidCellBase):
    features: int
    num_heads: int = 8
    head_dim: int = 16
    state_dim: int = 16
    conv_dim: int = 4
    kernel_init: Initializer = nn.initializers.lecun_normal()
    dtype: Dtype = jnp.float32
    param_dtype: Dtype = jnp.float32

    def setup(self):
        hidden_dim = self.num_heads * self.head_dim
        bc_dim = self.num_heads * self.state_dim
        conv_channels = hidden_dim + 2 * bc_dim

        self.A_log = self.param("A_log", _A_log_init(), (self.num_heads,))
        self.D = self.param("D", nn.initializers.ones, (self.num_heads,))
        self.dt_bias = self.param("dt_bias", _dt_bias_init(), (self.num_heads,))

        projection = partial(
            nn.Dense,
            kernel_init=self.kernel_init,
            use_bias=False,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )

        self.in_proj = projection(hidden_dim * 2)
        self.B = projection(bc_dim)
        self.C = projection(bc_dim)
        self.dt = projection(self.num_heads)
        self.conv = nn.Conv(
            conv_channels,
            kernel_size=(self.conv_dim,),
            padding=((self.conv_dim - 1, 0),),
            feature_group_count=conv_channels,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )
        self.norm = nn.RMSNorm(self.num_heads * self.head_dim)
        self.out_proj = nn.Dense(
            self.features,
            kernel_init=self.kernel_init,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )

    def _project(self, x: Array):
        batch_size, seq_len, _ = x.shape
        hidden_dim = self.num_heads * self.head_dim
        bc_dim = self.num_heads * self.state_dim

        proj = self.in_proj(x)
        u, z = jnp.split(proj, 2, axis=-1)

        B = self.B(x)
        C = self.C(x)

        xBC = jnp.concatenate([u, B, C], axis=-1)
        xBC = nn.silu(self.conv(xBC))

        u = xBC[..., :hidden_dim].reshape(
            batch_size, seq_len, self.num_heads, self.head_dim
        )
        B = xBC[..., hidden_dim : hidden_dim + bc_dim].reshape(
            batch_size, seq_len, self.num_heads, self.state_dim
        )
        C = xBC[..., hidden_dim + bc_dim :].reshape(
            batch_size, seq_len, self.num_heads, self.state_dim
        )

        dt = nn.softplus(self.dt(x) + self.dt_bias)

        return u, B, C, z, dt

    def __call__(self, x: Array, **kwargs) -> Carry:
        u, B, _, _, dt = self._project(x)

        A = -jnp.exp(self.A_log)
        decay = jnp.exp(dt * A[None, None, :])[:, :, :, None, None]

        state = jnp.einsum("bthn,bthd->bthnd", B * dt[..., None], u)

        return (state, decay)

    def binary_operator(self, a: Carry, b: Carry) -> Carry:
        state_i, decay_i = a
        state_j, decay_j = b
        return (decay_j * state_i + state_j, decay_j * decay_i)

    def read(self, h: Carry, x: Array, **kwargs) -> Array:
        batch_size, seq_len, _ = x.shape
        state, _ = h

        u, _, C, z, _ = self._project(x)

        y = jnp.einsum("bthn,bthnd->bthd", C, state)
        y = y + self.D[None, None, :, None] * u
        y = y.reshape(batch_size, seq_len, self.num_heads * self.head_dim)
        y = self.norm(y) * nn.silu(z)

        return self.out_proj(y)

    def initialize_carry(self, key, input_shape: Tuple[int, ...]) -> Carry:
        *batch_dims, _ = input_shape
        state = jnp.zeros(
            (*batch_dims, 1, self.num_heads, self.state_dim, self.head_dim),
            dtype=self.dtype,
        )
        decay = jnp.ones(
            (*batch_dims, 1, self.num_heads, 1, 1), dtype=self.dtype
        )
        return (state, decay)

    def local_jacobian(self, carry, z, inputs, **kwargs):
        B, T = inputs.shape[:2]
        NH, SD, HD = self.num_heads, self.state_dim, self.head_dim
        H = NH * SD * HD

        A = -jnp.exp(self.A_log)
        state_contrib = z[0]
        decay_5d = z[1]
        decay_ph = decay_5d[:, :, :, 0, 0]
        prev = carry[0]

        # Recover dt from decay: decay = exp(dt * A) → dt = log(decay) / A
        dt = jnp.log(jnp.maximum(decay_ph, 1e-30)) / A[None, None, :]
        sigma_dt = 1.0 - jnp.exp(-dt)

        # Flatten decay: (B,T,NH,1,1) → (B,T,NH*SD*HD)
        decay_flat = jnp.broadcast_to(
            decay_5d, (B, T, NH, SD, HD)
        ).reshape(B, T, H)

        # ∂decay/∂A_log = decay * dt * (-exp(A_log))
        dA_dAlog = -jnp.exp(self.A_log)
        J_Alog = (
            decay_5d * dt[:, :, :, None, None]
            * dA_dAlog[None, None, :, None, None] * prev
        ).reshape(B, T, H)

        # ∂h/∂dt_bias = σ_dt * (A * decay * prev + state_contrib / dt)
        dt_safe = jnp.maximum(dt, 1e-8)[:, :, :, None, None]
        J_dtbias = (
            sigma_dt[:, :, :, None, None]
            * (A[None, None, :, None, None] * decay_5d * prev
               + state_contrib / dt_safe)
        ).reshape(B, T, H)

        return decay_flat, {"A_log": J_Alog, "dt_bias": J_dtbias}

    def get_param_indices(self):
        H = self.num_heads * self.state_dim * self.head_dim
        idx = jnp.arange(H) // (self.state_dim * self.head_dim)
        return {"A_log": idx, "dt_bias": idx}

    def initialize_sensitivity(self, key, input_shape):
        *batch_dims, _ = input_shape
        H = self.num_heads * self.state_dim * self.head_dim
        z = jnp.zeros((*batch_dims, 1, H), dtype=self.dtype)
        return {"A_log": z, "dt_bias": z}
