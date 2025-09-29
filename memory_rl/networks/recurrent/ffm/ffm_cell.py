from functools import partial
import jax.numpy as jnp
from flax.linen.module import compact, nowrap
from flax.linen import initializers
from flax.linen.linear import Dense, default_kernel_init
from flax.linen.activation import sigmoid
from flax.linen.normalization import LayerNorm
from flax.typing import Array, PRNGKey, Dtype, Initializer

from flax.linen.recurrent import RNNCellBase


class FFMCell(RNNCellBase):
    """Fast & Forgetful Memory (FFM) cell."""

    features: int
    memory_size: int
    context_size: int

    min_period: int = 1
    max_period: int = 1024

    kernel_init: Initializer = default_kernel_init
    bias_init: Initializer = initializers.zeros_init()
    dtype: Dtype | None = None
    param_dtype: Dtype = jnp.float32

    @nowrap
    def initialize_carry(self, rng: PRNGKey, input_shape: tuple[int, ...]) -> Array:
        batch_dims = input_shape[:-1]
        mem_shape = batch_dims + (self.memory_size, self.context_size)
        complex_dtype = (
            jnp.complex64 if self.param_dtype == jnp.float32 else jnp.complex128
        )
        return jnp.zeros(mem_shape, dtype=complex_dtype)

    @property
    def num_feature_axes(self) -> int:
        return 1

    def _ab_constants(self) -> tuple[Array, Array]:
        a_var = self.variable(
            "constants",
            "a",
            lambda: jnp.linspace(
                -jnp.e, -1e-6, self.memory_size, dtype=self.param_dtype
            ),
        )
        b_var = self.variable(
            "constants",
            "b",
            lambda: (2 * jnp.pi)
            / jnp.linspace(
                float(self.min_period),
                float(self.max_period),
                self.context_size,
                dtype=self.param_dtype,
            ),
        )
        a = jnp.clip(a_var.value, a_max=jnp.array(-1e-6, dtype=self.param_dtype))
        b = b_var.value
        return a, b

    @compact
    def __call__(self, carry: Array, inputs: Array) -> tuple[Array, Array]:
        dense = partial(
            Dense,
            use_bias=True,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )
        pre = dense(features=self.memory_size, name="pre")
        gate_in_f = dense(features=self.memory_size, name="gate_in")
        gate_out_f = dense(features=self.features, name="gate_out")
        skip = dense(features=self.features, name="skip")
        mix = dense(features=self.features, name="mix")
        ln = LayerNorm(use_scale=False, use_bias=False, dtype=self.dtype, name="ln")

        gate_in = sigmoid(gate_in_f(inputs))
        x_mem = pre(inputs) * gate_in

        x_rep = jnp.repeat(x_mem[..., None], self.context_size, axis=-1)
        complex_dtype = (
            jnp.complex64 if self.param_dtype == jnp.float32 else jnp.complex128
        )
        x_c = x_rep.astype(complex_dtype)

        a, b = self._ab_constants()
        ab = a[:, None].astype(complex_dtype) + 1j * b[None, :].astype(complex_dtype)
        gamma_mc = jnp.exp(ab)
        gamma = gamma_mc.reshape(
            (1,) * (carry.ndim - 2) + (self.memory_size, self.context_size)
        )

        new_carry = carry * gamma + x_c

        z_in = jnp.concatenate([jnp.real(new_carry), jnp.imag(new_carry)], axis=-1)
        z_in = z_in.reshape(
            z_in.shape[:-2] + (self.memory_size * 2 * self.context_size,)
        )
        z = mix(z_in)

        g_out = sigmoid(gate_out_f(inputs))
        y = ln(z * g_out) + skip(inputs) * (1.0 - g_out)

        return new_carry, y
