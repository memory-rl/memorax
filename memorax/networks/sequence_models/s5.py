from typing import Any, Optional, Tuple

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.linen.recurrent import Carry
from jax.nn.initializers import lecun_normal, normal

from .sequence_model import SequenceModel
from .utils import (
    add_time_axis,
    discretize_bilinear,
    discretize_zoh,
    get_input_shape,
    init_cv,
    init_log_steps,
    init_v_inv_b,
    make_dplr_hippo,
    truncated_standard_normal,
)


class S5Layer(nn.Module):
    features: int
    state_size: int
    c_init: str
    discretization: str
    dt_min: float
    dt_max: float
    clip_eigens: bool
    step_rescale: float
    dtype: Any
    param_dtype: Any
    kernel_init: Any
    bias_init: Any

    def setup(self):
        lam, _, _, v, _ = make_dplr_hippo(self.state_size)
        self._lambda_real_init = jnp.asarray(lam.real, self.param_dtype)
        self._lambda_imag_init = jnp.asarray(lam.imag, self.param_dtype)
        self._v = v
        self._v_inv = v.conj().T

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
        a_out = (a_j * a_i) * (1.0 - c_j) + a_j * c_j
        b_out = (a_j * b_i + b_j) * (1.0 - c_j) + b_j * c_j
        c_out = c_i * (1.0 - c_j) + c_j
        return a_out, b_out, c_out

    @nn.compact
    def __call__(
        self,
        x: jax.Array,
        mask: jax.Array,
        carry: Carry,
    ):
        T, _ = x.shape
        lambda_bar, b_bar, c_tilde, d = self._discretized_params()
        a = jnp.broadcast_to(lambda_bar, (T, self.state_size))
        a = jnp.concatenate([jnp.ones((1, self.state_size)), a])

        b_x = jax.vmap(lambda x: b_bar @ x)(x)
        b_x = jnp.concatenate([carry, b_x])

        mask = jnp.concatenate([jnp.zeros(1), mask])
        mask = add_time_axis(mask)

        _, carry, _ = jax.lax.associative_scan(
            self._binary_operator_reset, (a, b_x, mask)
        )
        carry = carry[1:]

        ys = jax.vmap(lambda x: (c_tilde @ x).real)(carry)
        ys = ys + jax.vmap(lambda x: d * x)(x)
        return carry[jnp.newaxis, -1], ys


class S5Block(nn.Module):
    features: int
    state_size: int
    c_init: str
    discretization: str
    dt_min: float
    dt_max: float
    clip_eigens: bool
    step_rescale: float
    dtype: Any
    param_dtype: Any
    kernel_init: Any
    bias_init: Any

    @nn.compact
    def __call__(
        self,
        x: jax.Array,
        mask: jax.Array,
        carry: Carry,
    ):
        skip = x
        x = nn.LayerNorm()(x)
        carry, x = jax.vmap(
            S5Layer(
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
            )
        )(x, mask, carry)
        x = nn.LayerNorm()(x)
        x = nn.gelu(x)
        x = x + skip
        return carry, x


class S5(SequenceModel, nn.Module):
    state_size: int
    num_layers: int
    c_init: str = "truncated_standard_normal"
    discretization: str = "zoh"
    dt_min: float = 0.001
    dt_max: float = 0.1
    clip_eigenvectors: bool = False
    step_rescale: float = 1.0
    dtype: Any = jnp.float32
    param_dtype: Any = jnp.float32
    kernel_init: Any = None
    bias_init: Any = None

    def initialize_carry(self, key: jax.Array, input_shape: Tuple[int, ...]) -> tuple:
        batch_size, *_ = input_shape

        initial_carry = jnp.zeros(
            (
                batch_size,
                1,
                self.state_size,
            ),
            dtype=jnp.complex64,
        )
        return tuple(initial_carry for _ in range(self.num_layers))

    @nn.compact
    def __call__(
        self,
        inputs: jax.Array,
        mask: jax.Array,
        initial_carry: Optional[Carry] = None,
        **kwargs,
    ):
        new_carry = []

        if initial_carry is None:
            input_shape = get_input_shape(inputs)
            initial_carry = self.initialize_carry(jax.random.key(0), input_shape)

        x = inputs
        mask = mask.astype(jnp.float32)
        for carry_i in initial_carry:
            new_carry_i, x = S5Block(
                features=self.features,
                state_size=self.state_size,
                c_init=self.c_init,
                discretization=self.discretization,
                dt_min=self.dt_min,
                dt_max=self.dt_max,
                clip_eigens=self.clip_eigenvectors,
                step_rescale=self.step_rescale,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                kernel_init=self.kernel_init,
                bias_init=self.bias_init,
            )(x, mask, carry_i)
            new_carry.append(new_carry_i)
        new_carry = tuple(new_carry)
        return new_carry, x
