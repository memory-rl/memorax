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
            jax.random.uniform(key, shape) * (jnp.log(dt_max) - jnp.log(dt_min))
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
        state_projection_dim = self.num_heads * self.state_dim
        conv_channels = hidden_dim + 2 * state_projection_dim

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

        self.input_projection = projection(hidden_dim * 2)
        self.B = projection(state_projection_dim)
        self.C = projection(state_projection_dim)
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
        self.output_projection = nn.Dense(
            self.features,
            kernel_init=self.kernel_init,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )

    def _project(self, x: Array):
        B, T, _ = x.shape
        hidden_dim = self.num_heads * self.head_dim
        state_projection_dim = self.num_heads * self.state_dim

        proj = self.input_projection(x)
        hidden, z = jnp.split(proj, 2, axis=-1)

        B_proj = self.B(x)
        C_proj = self.C(x)

        conv_input = jnp.concatenate([hidden, B_proj, C_proj], axis=-1)
        conv_input = nn.silu(self.conv(conv_input))

        hidden = conv_input[..., :hidden_dim].reshape(
            B, T, self.num_heads, self.head_dim
        )
        B_proj = conv_input[
            ..., hidden_dim : hidden_dim + state_projection_dim
        ].reshape(B, T, self.num_heads, self.state_dim)
        C_proj = conv_input[..., hidden_dim + state_projection_dim :].reshape(
            B, T, self.num_heads, self.state_dim
        )

        dt = nn.softplus(self.dt(x) + self.dt_bias)

        return hidden, B_proj, C_proj, z, dt

    def __call__(self, x: Array, **kwargs) -> Carry:
        hidden, B_proj, _, _, dt = self._project(x)

        A = -jnp.exp(self.A_log)
        decay = jnp.exp(dt * A[None, None, :])[:, :, :, None, None]

        h = jnp.einsum("bthn,bthd->bthnd", B_proj * dt[..., None], hidden)

        return (h, decay)

    def binary_operator(self, a: Carry, b: Carry) -> Carry:
        h_i, decay_i = a
        h_j, decay_j = b
        return (decay_j * h_i + h_j, decay_j * decay_i)

    def read(self, carry: Carry, x: Array, **kwargs) -> Array:
        B, T, _ = x.shape
        h, _ = carry

        hidden, _, C_proj, z, _ = self._project(x)

        y = jnp.einsum("bthn,bthnd->bthd", C_proj, h)
        y = y + self.D[None, None, :, None] * hidden
        y = y.reshape(B, T, self.num_heads * self.head_dim)
        y = self.norm(y) * nn.silu(z)

        return self.output_projection(y)

    def initialize_carry(self, key: jax.Array, input_shape: Tuple[int, ...]) -> Carry:
        *batch_dims, _ = input_shape
        h = jnp.zeros(
            (*batch_dims, 1, self.num_heads, self.state_dim, self.head_dim),
            dtype=self.dtype,
        )
        decay = jnp.ones((*batch_dims, 1, self.num_heads, 1, 1), dtype=self.dtype)
        return (h, decay)

    def local_jacobian(self, carry, z, inputs, **kwargs):
        B, T = inputs.shape[:2]
        num_heads, state_dim, head_dim = self.num_heads, self.state_dim, self.head_dim
        H = num_heads * state_dim * head_dim

        A = -jnp.exp(self.A_log)
        h_t = z[0]
        decay = z[1]
        decay_per_head = decay[:, :, :, 0, 0]
        h_prev = carry[0]

        # Recover dt from decay: decay = exp(dt * A) → dt = log(decay) / A
        dt = jnp.log(jnp.maximum(decay_per_head, 1e-30)) / A[None, None, :]
        sigmoid_dt = 1.0 - jnp.exp(-dt)
        dA_dA_log = -jnp.exp(self.A_log)
        dt_safe = jnp.maximum(dt, 1e-8)

        # Expand per-head scalars to 5D for broadcasting with (B,T,NH,SD,HD)
        dt = dt[:, :, :, None, None]
        dt_safe = dt_safe[:, :, :, None, None]
        sigmoid_dt = sigmoid_dt[:, :, :, None, None]
        A = A[None, None, :, None, None]
        dA_dA_log = dA_dA_log[None, None, :, None, None]

        # Flatten decay: (B,T,NH,1,1) → (B,T,NH*SD*HD)
        decay_flat = jnp.broadcast_to(
            decay, (B, T, num_heads, state_dim, head_dim)
        ).reshape(B, T, H)

        # ∂h/∂A_log = decay * dt * (∂A/∂A_log) * h_prev
        J_A_log = (decay * dt * dA_dA_log * h_prev).reshape(B, T, H)

        # ∂h/∂dt_bias = sigmoid_dt * (A * decay * h_prev + h_t / dt)
        J_dt_bias = (sigmoid_dt * (A * decay * h_prev + h_t / dt_safe)).reshape(B, T, H)

        return decay_flat, {"A_log": J_A_log, "dt_bias": J_dt_bias}

    def get_param_indices(self):
        H = self.num_heads * self.state_dim * self.head_dim
        idx = jnp.arange(H) // (self.state_dim * self.head_dim)
        return {"A_log": idx, "dt_bias": idx}

    def initialize_sensitivity(self, key, input_shape):
        *batch_dims, _ = input_shape
        H = self.num_heads * self.state_dim * self.head_dim
        zeros = jnp.zeros((*batch_dims, 1, H), dtype=self.dtype)
        return {"A_log": zeros, "dt_bias": zeros}
