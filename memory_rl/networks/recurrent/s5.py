from typing import Any, Tuple
from flax.linen.recurrent import Carry
import jax
import jax.numpy as jnp
from flax import linen as nn
from jax.nn.initializers import lecun_normal, normal

from .utils import (
    discretize_bilinear,
    discretize_zoh,
    init_cv,
    init_log_steps,
    init_v_inv_b,
    make_dplr_hippo,
    truncated_standard_normal,
)


class S5Layer(nn.Module):
    features: int
    state_size: int
    c_init: str = "truncated_standard_normal"
    discretization: str = "zoh"
    dt_min: float = 0.001
    dt_max: float = 0.1
    clip_eigens: bool = False
    step_rescale: float = 1.0
    dtype: Any = jnp.float32
    param_dtype: Any = jnp.float32
    kernel_init: Any = None
    bias_init: Any = None

    def setup(self):
        lam, _, _, v, _ = make_dplr_hippo(self.state_size)
        self._lambda_real_init = jnp.asarray(lam.real, self.param_dtype)
        self._lambda_imag_init = jnp.asarray(lam.imag, self.param_dtype)
        self._v = v
        self._v_inv = v.conj().T

    @property
    def num_feature_axes(self) -> int:
        return 1

    def _discretized_params(self):
        lambda_real = self.param(
            "lambda_real", lambda rng, shape: self._lambda_real_init, (self.state_size,)
        )
        lambda_imag = self.param(
            "lambda_imag", lambda rng, shape: self._lambda_imag_init, (self.state_size,)
        )
        if self.clip_eigens:
            lambda_real = jnp.minimum(lambda_real, -1e-4)
            lam = jax.lax.complex(lambda_real, lambda_imag)
        else:
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
                raise ValueError("invalid c_init")

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
                raise ValueError("invalid discretization")
        return lambda_bar, b_bar, c_tilde, d.astype(self.dtype)

    @staticmethod
    def _binary_operator_reset(q_i, q_j):
        a_i, b_i, c_i = q_i
        a_j, b_j, c_j = q_j
        c_j = c_j[..., None]
        a_out = (a_j * a_i) * (1.0 - c_j) + a_j * c_j
        b_out = (a_j * b_i + b_j) * (1.0 - c_j) + b_j * c_j
        c_out = c_i * (1.0 - c_j[..., 0]) + c_j[..., 0]
        return a_out, b_out, c_out

    @nn.compact
    def __call__(
        self,
        inputs: jax.Array,
        mask: jax.Array,
        carry: Carry,
    ):
        u = jnp.swapaxes(inputs, 0, 1)
        mask = jnp.swapaxes(mask, 0, 1)
        lambda_bar, b_bar, c_tilde, d = self._discretized_params()
        t, batch_dims, h = u.shape
        a = jnp.broadcast_to(
            lambda_bar[None, None, :], (t, u.shape[1], self.state_size)
        )
        a = jnp.concatenate(
            [jnp.ones((1, u.shape[1], self.state_size), a.dtype), a], axis=0
        )
        bu = jnp.einsum("ph,tbh->tbp", b_bar, u).astype(jnp.complex64)
        bu = jnp.concatenate([carry[None, ...], bu], axis=0)
        mask = jnp.concatenate([jnp.zeros((1, u.shape[1]), mask.dtype), mask], axis=0)
        _, xs, _ = jax.lax.associative_scan(self._binary_operator_reset, (a, bu, mask))
        xs = xs[1:]
        y = jnp.einsum("hp,tbp->tbh", c_tilde, xs).real
        y = y + jnp.einsum("h,tbh->tbh", d, u)
        y = y.reshape((t, batch_dims, h)).astype(self.dtype)
        x_t = xs[-1].reshape((batch_dims, self.state_size))

        y = jnp.swapaxes(y, 0, 1)
        return x_t, y


class S5(nn.Module):
    features: int
    state_size: int
    num_layers: int
    c_init: str = "truncated_standard_normal"
    discretization: str = "zoh"
    dt_min: float = 0.001
    dt_max: float = 0.1
    clip_eigens: bool = False
    step_rescale: float = 1.0
    dtype: Any = jnp.float32
    param_dtype: Any = jnp.float32
    kernel_init: Any = None
    bias_init: Any = None

    def initialize_carry(self, rng: jax.Array, input_shape: Tuple[int, ...]) -> tuple:
        batch_dims = input_shape[:-1]
        return tuple(
            jnp.zeros(batch_dims + (self.state_size,), dtype=jnp.complex64)
            for _ in range(self.num_layers)
        )

    @nn.compact
    def __call__(
        self,
        inputs: jax.Array,
        mask: jax.Array,
        initial_carry: Carry,
    ):
        new_carry = []

        u = inputs
        mask = mask.astype(jnp.float32)
        for carry_i in initial_carry:
            new_carry_i, u = S5Layer(
                features=self.features,
                state_size=self.state_size,
                c_init=self.c_init,
                discretization=self.discretization,
                dt_min=self.dt_min,
                dt_max=self.dt_max,
                clip_eigens=self.clip_eigens,
                step_rescale=self.step_rescale,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                kernel_init=self.kernel_init,
                bias_init=self.bias_init,
            )(u, mask, carry_i)
            new_carry.append(new_carry_i)
        new_carry = tuple(new_carry)
        return new_carry, u
