from functools import partial
from typing import Optional, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.linen import initializers
from flax.linen.linear import Dense, default_kernel_init
from flax.linen.normalization import LayerNorm
from flax.typing import Dtype, Initializer
from memorax.utils.typing import Array, Carry

from .memoroid import MemoroidCellBase


class FFMCell(MemoroidCellBase):
    """Fast and Forgetful Memory algebra.

    Uses position-relative decay with complex exponential basis functions
    for long-range dependencies.

    Element: (state, timestep)
    Combine: (state_i * γ(t_j - t_i) + state_j, t_j)

    The decay γ(Δt) = exp((a + ib) * Δt) where a controls decay rate
    and b controls oscillation frequency.
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
    dtype: Dtype | None = None
    param_dtype: Dtype = jnp.float32

    def setup(self) -> None:
        self.limit = (
            jnp.log(jnp.finfo(self.param_dtype).max) / self.max_period - self.epsilon
        )

        low = -self.limit + self.epsilon
        high = jnp.maximum(
            jnp.minimum(-1e-6, jnp.log(self.beta) / self.max_period), low
        )
        self.a = self.param(
            "a",
            lambda _: jnp.linspace(low, high, self.memory_size, dtype=self.param_dtype),
        )
        self.b = self.param(
            "b",
            lambda _: (2 * jnp.pi)
            / jnp.linspace(
                self.min_period,
                self.max_period,
                self.context_size,
                dtype=self.param_dtype,
            ),
        )

    def _complex_dtype(self):
        return jnp.complex64 if self.param_dtype == jnp.float32 else jnp.complex128

    def _gamma(self, delta_t):
        """Compute decay factor for time difference delta_t."""
        a = jnp.clip(self.a, min=-self.limit, max=-1e-8)
        a = a.reshape(1, self.memory_size, 1)
        b = self.b.reshape(1, 1, self.context_size)
        ab = jax.lax.complex(a, b)
        return jnp.exp(ab * delta_t[..., None, None])

    @nn.compact
    def __call__(self, inputs: Array, **kwargs) -> Carry:
        B, T, _ = inputs.shape

        dense = partial(
            Dense,
            use_bias=True,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )

        # Input projections
        pre = dense(features=self.memory_size, name="pre")(inputs)
        input_gate = nn.sigmoid(
            dense(features=self.memory_size, name="input_gate")(inputs)
        )
        x = pre * input_gate

        # Expand to context size and make complex
        state = jnp.repeat(x[..., None], self.context_size, axis=-1)
        state = jax.lax.complex(state, jnp.zeros_like(state))

        # Timesteps (complex for compatibility)
        timestep = jnp.arange(T, dtype=self.param_dtype)
        timestep = jnp.broadcast_to(timestep, (B, T))
        timestep = jax.lax.complex(timestep, jnp.zeros_like(timestep))

        return (state, timestep)

    def binary_operator(self, a: Carry, b: Carry) -> Carry:
        """Position-relative combine: state_i * γ(t_j - t_i) + state_j"""
        state_i, t_i = a
        state_j, t_j = b
        delta_t = t_j - t_i
        gamma = self._gamma(delta_t)
        return (state_i * gamma + state_j, t_j)

    @nn.compact
    def read(self, h: Carry, x: Array, **kwargs) -> Array:
        state, _ = h

        dense = partial(
            Dense,
            use_bias=True,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )
        ln = LayerNorm(use_scale=False, use_bias=False, name="ln")

        # Output projections
        output_gate = nn.sigmoid(dense(features=self.features, name="output_gate")(x))
        skip = dense(features=self.features, name="skip")(x)

        # Mix real and imaginary parts
        z_in = jnp.concatenate([jnp.real(state), jnp.imag(state)], axis=-1)
        z_in = z_in.reshape((*z_in.shape[:-2], -1))
        z = dense(features=self.features, name="mix")(z_in)

        y = ln(z) * output_gate + skip * (1.0 - output_gate)
        return y

    def initialize_carry(self, key: jax.Array, input_shape: Tuple[int, ...]) -> Carry:
        batch_size, *_ = input_shape
        state = jnp.zeros(
            (batch_size, 1, self.memory_size, self.context_size),
            dtype=self._complex_dtype(),
        )
        # Start at timestep -1 so first real timestep is 0
        timestep = jnp.full((batch_size, 1), -1, dtype=self._complex_dtype())
        return (state, timestep)
