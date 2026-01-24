from typing import Tuple

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.typing import Dtype
from jax.nn.initializers import lecun_normal, normal
from memorax.utils.typing import Array, Carry

from .memoroid import Algebra
from .utils import (
    discretize_bilinear,
    discretize_zoh,
    init_cv,
    init_log_steps,
    init_v_inv_b,
    make_dplr_hippo,
    truncated_standard_normal,
)


class S5(Algebra):
    """S5 (Structured State Space for Sequences) algebra.

    Uses HIPPO matrix initialization and discretization for stable
    long-range dependencies.

    Element: (decay, state) where state accumulates B @ x
    Combine: (a_j * a_i, a_j * s_i + s_j)
    """

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
        self._lambda_real_init = jnp.asarray(lam.real, self.param_dtype)
        self._lambda_imag_init = jnp.asarray(lam.imag, self.param_dtype)
        self._v = v
        self._v_inv = v.conj().T

    def _discretized_params(self):
        lambda_real = self.param(
            "lambda_real",
            lambda rng, shape: self._lambda_real_init,
            (self.state_size,),
        )
        lambda_imag = self.param(
            "lambda_imag",
            lambda rng, shape: self._lambda_imag_init,
            (self.state_size,),
        )

        if self.clip_eigens:
            lambda_real = jnp.minimum(lambda_real, -1e-4)

        lam = jax.lax.complex(lambda_real, lambda_imag)

        b = self.param(
            "b",
            lambda rng, shape: init_v_inv_b(
                lecun_normal(), rng, (self.state_size, self.features), self._v_inv
            ),
            (self.state_size, self.features, 2),
        )
        b_tilde = jax.lax.complex(b[..., 0], b[..., 1])

        match self.c_init:
            case "complex_normal":
                c = self.param(
                    "c", normal(stddev=0.5**0.5), (self.features, self.state_size, 2)
                )
                c_tilde = jax.lax.complex(c[..., 0], c[..., 1])
            case "lecun_normal":
                c = self.param(
                    "c",
                    lambda rng, shape: init_cv(
                        lecun_normal(),
                        rng,
                        (self.features, self.state_size, 2),
                        self._v,
                    ),
                    (self.features, self.state_size, 2),
                )
                c_tilde = jax.lax.complex(c[..., 0], c[..., 1])
            case "truncated_standard_normal":
                c = self.param(
                    "c",
                    lambda rng, shape: init_cv(
                        truncated_standard_normal,
                        rng,
                        (self.features, self.state_size, 2),
                        self._v,
                    ),
                    (self.features, self.state_size, 2),
                )
                c_tilde = jax.lax.complex(c[..., 0], c[..., 1])
            case _:
                raise ValueError(f"Invalid c_init: {self.c_init}")

        d = self.param("d", normal(stddev=1.0), (self.features,))

        log_step = self.param(
            "log_step", init_log_steps, (self.state_size, self.dt_min, self.dt_max)
        )
        step = self.step_rescale * jnp.exp(log_step[:, 0].astype(jnp.float32))

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

        return lambda_bar, b_bar, c_tilde, d.astype(self.dtype)

    @nn.compact
    def __call__(self, x: Array, **kwargs) -> Carry:
        B, T, _ = x.shape
        lambda_bar, b_bar, _, _ = self._discretized_params()

        # Decay: broadcast lambda to (B, T, state_size)
        decay = jnp.broadcast_to(lambda_bar, (B, T, self.state_size))

        # State: B @ x for each timestep
        state = jax.vmap(jax.vmap(lambda xi: b_bar @ xi))(x)

        return (decay, state)

    def combine(self, a: Carry, b: Carry) -> Carry:
        """Diagonal SSM combine: (a_j * a_i, a_j * s_i + s_j)"""
        decay_i, state_i = a
        decay_j, state_j = b
        return (decay_j * decay_i, decay_j * state_i + state_j)

    @nn.compact
    def read(self, h: Carry, x: Array, **kwargs) -> Array:
        _, _, c_tilde, d = self._discretized_params()
        _, state = h

        # Output: C @ state + D * x
        y = jax.vmap(jax.vmap(lambda si, xi: (c_tilde @ si).real + d * xi))(state, x)
        return y

    def initialize_carry(self, key: jax.Array, input_shape: Tuple[int, ...]) -> Carry:
        batch_size, *_ = input_shape
        decay = jnp.ones((batch_size, 1, self.state_size), dtype=jnp.complex64)
        state = jnp.zeros((batch_size, 1, self.state_size), dtype=jnp.complex64)
        return (decay, state)
