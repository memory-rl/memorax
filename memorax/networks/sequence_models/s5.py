from functools import partial
from typing import Tuple

import jax
import jax.numpy as jnp
from flax.typing import Dtype
from jax.nn.initializers import lecun_normal, normal

from memorax.utils.typing import Array, Carry

from memorax.networks.hippo import discretize_bilinear, discretize_zoh, make_dplr_hippo
from memorax.networks.initializers import init_cv, init_v_inv_b, log_step, truncated_standard_normal

from .memoroid import MemoroidCellBase


def _c_init_complex_normal(key, shape):
    return normal(stddev=0.5**0.5)(key, shape)


def _c_init_lecun(key, shape, v):
    return init_cv(lecun_normal(), key, shape, v)


def _c_init_truncated(key, shape, v):
    return init_cv(truncated_standard_normal, key, shape, v)


class S5Cell(MemoroidCellBase):
    features: int
    state_size: int
    c_init: str = "truncated_standard_normal"
    discretization: str = "zoh"
    dt_min: float = 0.001
    dt_max: float = 0.1
    clip_eigens: bool = False
    step_rescale: float = 1.0
    dtype: Dtype = jnp.float32
    param_dtype: Dtype = jnp.float32

    def setup(self):
        lam, _, _, v, _ = make_dplr_hippo(self.state_size)
        lambda_real_init = jnp.asarray(lam.real, self.param_dtype)
        lambda_imag_init = jnp.asarray(lam.imag, self.param_dtype)
        self._v = v
        self._v_inv = v.conj().T

        self.lambda_real = self.param(
            "lambda_real",
            lambda rng, shape: lambda_real_init,
            (self.state_size,),
        )
        self.lambda_imag = self.param(
            "lambda_imag",
            lambda rng, shape: lambda_imag_init,
            (self.state_size,),
        )
        self.B = self.param(
            "b",
            lambda rng, shape: init_v_inv_b(
                lecun_normal(), rng, (self.state_size, self.features), self._v_inv
            ),
            (self.state_size, self.features, 2),
        )

        match self.c_init:
            case "complex_normal":
                self.C = self.param(
                    "c",
                    _c_init_complex_normal,
                    (self.features, self.state_size, 2),
                )
            case "lecun_normal":
                self.C = self.param(
                    "c",
                    partial(_c_init_lecun, v=self._v),
                    (self.features, self.state_size, 2),
                )
            case "truncated_standard_normal":
                self.C = self.param(
                    "c",
                    partial(_c_init_truncated, v=self._v),
                    (self.features, self.state_size, 2),
                )
            case _:
                raise ValueError(f"Invalid c_init: {self.c_init}")

        self.d = self.param("d", normal(stddev=1.0), (self.features,))
        self.log_step = self.param(
            "log_step", log_step(self.dt_min, self.dt_max), (self.state_size,)
        )

    def _discretized_params(self):
        lambda_real = self.lambda_real
        lambda_imag = self.lambda_imag

        if self.clip_eigens:
            lambda_real = jnp.minimum(lambda_real, -1e-4)

        lam = jax.lax.complex(lambda_real, lambda_imag)
        b_tilde = jax.lax.complex(self.B[..., 0], self.B[..., 1])
        c_tilde = jax.lax.complex(self.C[..., 0], self.C[..., 1])
        step = self.step_rescale * jnp.exp(self.log_step[:, 0].astype(jnp.float32))

        match self.discretization:
            case "zoh":
                lambda_bar, b_bar = discretize_zoh(
                    lam.astype(jnp.complex64),
                    b_tilde.astype(jnp.complex64),
                    step.astype(jnp.complex64),
                )
            case "bilinear":
                lambda_bar, b_bar = discretize_bilinear(
                    lam.astype(jnp.complex64),
                    b_tilde.astype(jnp.complex64),
                    step.astype(jnp.complex64),
                )
            case _:
                raise ValueError(f"Invalid discretization: {self.discretization}")

        return lambda_bar, b_bar, c_tilde, self.d.astype(self.dtype), lam, b_tilde, step

    def __call__(self, x: Array, **kwargs) -> Carry:
        B, T, _ = x.shape
        lambda_bar, b_bar, _, _, _, _, _ = self._discretized_params()

        decay = jnp.broadcast_to(lambda_bar, (B, T, self.state_size))

        state = jax.vmap(jax.vmap(lambda xi: b_bar @ xi))(x)

        return (state, decay)

    def binary_operator(self, a: Carry, b: Carry) -> Carry:
        state_i, decay_i = a
        state_j, decay_j = b
        return (decay_j * state_i + state_j, decay_j * decay_i)

    def read(self, h: Carry, x: Array, **kwargs) -> Array:
        _, _, c_tilde, d, _, _, _ = self._discretized_params()
        state, _ = h

        y = jax.vmap(jax.vmap(lambda si, xi: (c_tilde @ si).real + d * xi))(state, x)
        return y

    def initialize_carry(self, key: jax.Array, input_shape: Tuple[int, ...]) -> Carry:
        *batch_dims, _ = input_shape
        state = jnp.zeros((*batch_dims, 1, self.state_size), dtype=jnp.complex64)
        decay = jnp.ones((*batch_dims, 1, self.state_size), dtype=jnp.complex64)
        return (state, decay)

