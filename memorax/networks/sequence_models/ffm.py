from functools import partial
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.linen import initializers
from flax.linen.linear import default_kernel_init
from flax.typing import Dtype, Initializer

from memorax.utils.typing import Array, Carry

from .memoroid import MemoroidCellBase


class FFMCell(MemoroidCellBase):
    """Fast and Forgetful Memory algebra.

    Uses position-relative decay with complex exponential basis functions
    for long-range dependencies.

    Element: (S, t)
    Combine: (S_i * γ(t_j - t_i) + S_j, t_j)

    The decay γ(Δt) = exp((α + iω) * Δt) where α controls decay rate
    and ω controls oscillation frequency.
    """

    features: int
    memory_size: int
    context_size: int
    min_period: int = 1
    max_period: int = 1024
    epsilon: float = 0.01
    beta: float = 0.01
    kernel_init: Initializer = default_kernel_init
    bias_init: Initializer = initializers.zeros_init()
    dtype: Optional[Dtype] = None
    param_dtype: Dtype = jnp.float32

    def setup(self):
        self.limit = (
            jnp.log(jnp.finfo(self.param_dtype).max) / self.max_period - self.epsilon
        )

        low = -self.limit + self.epsilon
        high = jnp.maximum(
            jnp.minimum(-1e-6, jnp.log(self.beta) / self.max_period), low
        )
        self.alpha = self.param(
            "alpha",
            lambda _: jnp.linspace(low, high, self.memory_size, dtype=self.param_dtype),
        )
        self.omega = self.param(
            "omega",
            lambda _: (2 * jnp.pi)
            / jnp.linspace(
                self.min_period,
                self.max_period,
                self.context_size,
                dtype=self.param_dtype,
            ),
        )

        dense = partial(
            nn.Dense,
            use_bias=True,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )

        self.pre = dense(features=self.memory_size, name="pre")
        self.input_gate = dense(features=self.memory_size, name="input_gate")
        self.output_gate = dense(features=self.features, name="output_gate")
        self.skip = dense(features=self.features, name="skip")
        self.mix = dense(features=self.features, name="mix")
        self.ln = nn.LayerNorm(use_scale=False, use_bias=False, name="ln")

    def _complex_dtype(self):
        return jnp.complex64 if self.param_dtype == jnp.float32 else jnp.complex128

    def _gamma(self, dt):
        """Compute decay factor γ(Δt) = exp((α + iω) * Δt)."""
        alpha = jnp.clip(self.alpha, min=-self.limit, max=-1e-8)
        alpha = alpha.reshape(1, self.memory_size, 1)
        omega = self.omega.reshape(1, 1, self.context_size)
        return jnp.exp(jax.lax.complex(alpha, omega) * dt[..., None, None])

    def __call__(self, x: Array, **kwargs) -> Carry:
        B, T, _ = x.shape

        pre = self.pre(x)
        gate = nn.sigmoid(self.input_gate(x))
        x_tilde = pre * gate

        S = jnp.repeat(x_tilde[..., None], self.context_size, axis=-1)
        S = jax.lax.complex(S, jnp.zeros_like(S))

        t = jnp.arange(T, dtype=self.param_dtype)
        t = jnp.broadcast_to(t, (B, T))
        t = jax.lax.complex(t, jnp.zeros_like(t))

        return (S, t)

    def binary_operator(self, a: Carry, b: Carry) -> Carry:
        """Position-relative combine: S_i * γ(t_j - t_i) + S_j"""
        S_i, t_i = a
        S_j, t_j = b
        dt = t_j - t_i
        gamma = self._gamma(dt)
        return (S_i * gamma + S_j, t_j)

    def read(self, h: Carry, x: Array, **kwargs) -> Array:
        S, _ = h

        output_gate = nn.sigmoid(self.output_gate(x))
        skip = self.skip(x)

        z = jnp.concatenate([jnp.real(S), jnp.imag(S)], axis=-1)
        z = z.reshape((*z.shape[:-2], -1))
        z = self.mix(z)

        y = self.ln(z) * output_gate + skip * (1.0 - output_gate)
        return y

    def initialize_carry(self, key: jax.Array, input_shape: Tuple[int, ...]) -> Carry:
        *batch_dims, _ = input_shape
        S = jnp.zeros(
            (*batch_dims, 1, self.memory_size, self.context_size),
            dtype=self._complex_dtype(),
        )
        t = jnp.full((*batch_dims, 1), -1, dtype=self._complex_dtype())
        return (S, t)

    def local_jacobian(self, carry, z, inputs, **kwargs):
        B, T = inputs.shape[:2]
        H = self.memory_size * self.context_size

        alpha = jnp.clip(self.alpha, min=-self.limit, max=-1e-8)
        gamma = jnp.exp(jax.lax.complex(alpha[:, None], self.omega[None, :])).reshape(
            1, 1, H
        )

        decay = jnp.broadcast_to(gamma, (B, T, H))

        S = carry[0].reshape(B, T, H)
        # J_alpha[h] = ∂S[h]/∂α[h//cs] = γ[h] * S[h]
        J_alpha = gamma * S
        # J_omega[h] = ∂S[h]/∂ω[h%cs] = iγ[h] * S[h]
        J_omega = 1j * gamma * S

        return decay, {"alpha": J_alpha, "omega": J_omega}

    def get_param_indices(self):
        H = self.memory_size * self.context_size
        return {
            "alpha": jnp.arange(H) // self.context_size,
            "omega": jnp.arange(H) % self.context_size,
        }

    def initialize_sensitivity(self, key, input_shape):
        *batch_dims, _ = input_shape
        H = self.memory_size * self.context_size
        zeros = jnp.zeros((*batch_dims, 1, H), dtype=self._complex_dtype())
        return {"alpha": zeros, "omega": zeros}
