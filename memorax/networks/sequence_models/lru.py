from functools import partial
from typing import Tuple

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.typing import Dtype
from memorax.utils.typing import Array, Carry

from .memoroid import MemoroidCellBase


def _nu_init(key, shape, r_min, r_max, dtype=jnp.float32):
    u = jax.random.uniform(key=key, shape=shape, dtype=dtype)
    return jnp.log(-0.5 * jnp.log(u * (r_max**2 - r_min**2) + r_min**2))


def _theta_init(key, shape, max_phase, dtype=jnp.float32):
    u = jax.random.uniform(key, shape=shape, dtype=dtype)
    return jnp.log(max_phase * u)


def _gamma_log_init(key, lamb):
    nu, theta = lamb
    diag_lambda = jnp.exp(-jnp.exp(nu) + 1j * jnp.exp(theta))
    return jnp.log(jnp.sqrt(1 - jnp.abs(diag_lambda) ** 2))


def _matrix_init(key, shape, dtype=jnp.float32, normalization=1):
    return jax.random.normal(key=key, shape=shape, dtype=dtype) / normalization


class LRUCell(MemoroidCellBase):
    """Linear Recurrent Unit algebra.

    Uses exponential parameterization of eigenvalues for stable training.

    Element: (decay, state)
    Combine: (a_j * a_i, a_j * s_i + s_j)
    """

    features: int
    hidden_dim: int
    r_min: float = 0.0
    r_max: float = 1.0
    max_phase: float = 6.28
    dtype: Dtype = jnp.float32
    param_dtype: Dtype = jnp.float32

    def setup(self):
        self.theta_log = self.param(
            "theta_log",
            partial(_theta_init, max_phase=self.max_phase),
            (self.hidden_dim,),
        )
        self.nu_log = self.param(
            "nu_log",
            partial(_nu_init, r_min=self.r_min, r_max=self.r_max),
            (self.hidden_dim,),
        )
        self.gamma_log = self.param(
            "gamma_log", _gamma_log_init, (self.nu_log, self.theta_log)
        )

        self.B_real = self.param(
            "B_real",
            partial(_matrix_init, normalization=jnp.sqrt(2 * self.features)),
            (self.hidden_dim, self.features),
        )
        self.B_imag = self.param(
            "B_imag",
            partial(_matrix_init, normalization=jnp.sqrt(2 * self.features)),
            (self.hidden_dim, self.features),
        )
        self.C_real = self.param(
            "C_real",
            partial(_matrix_init, normalization=jnp.sqrt(self.hidden_dim)),
            (self.features, self.hidden_dim),
        )
        self.C_imag = self.param(
            "C_imag",
            partial(_matrix_init, normalization=jnp.sqrt(self.hidden_dim)),
            (self.features, self.hidden_dim),
        )
        self.D = self.param("D", _matrix_init, (self.features,))

    def __call__(self, x: Array, **kwargs) -> Carry:
        B, T, _ = x.shape

        diag_lambda = jnp.exp(-jnp.exp(self.nu_log) + 1j * jnp.exp(self.theta_log))
        B_norm = (self.B_real + 1j * self.B_imag) * jnp.expand_dims(
            jnp.exp(self.gamma_log), axis=-1
        )

        decay = jnp.broadcast_to(diag_lambda, (B, T, self.hidden_dim))

        # State: B @ x for each timestep
        state = jax.vmap(jax.vmap(lambda u: B_norm @ u))(x)

        return (decay, state)

    def binary_operator(self, a: Carry, b: Carry) -> Carry:
        """Diagonal SSM combine: (a_j * a_i, a_j * s_i + s_j)"""
        decay_i, state_i = a
        decay_j, state_j = b
        return (decay_j * decay_i, decay_j * state_i + state_j)

    def read(self, h: Carry, x: Array, **kwargs) -> Array:
        C = jax.lax.complex(self.C_real, self.C_imag)
        _, state = h

        # Output: C @ state + D * x
        y = jax.vmap(jax.vmap(lambda si, xi: (C @ si).real + self.D * xi))(state, x)
        return y

    def initialize_carry(self, key: jax.Array, input_shape: Tuple[int, ...]) -> Carry:
        batch_size, *_ = input_shape
        decay = jnp.ones((batch_size, 1, self.hidden_dim), dtype=jnp.complex64)
        state = jnp.zeros((batch_size, 1, self.hidden_dim), dtype=jnp.complex64)
        return (decay, state)
