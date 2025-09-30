from functools import partial
import jax
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

    def _complex_dtype_from(self, param_dtype):
        return jnp.complex64 if param_dtype == jnp.float32 else jnp.complex128

    @nowrap
    def initialize_carry(self, rng: PRNGKey, input: tuple[int, ...]) -> Array:
        batch_dims = input[:-1]
        mem_shape = batch_dims + (self.memory_size, self.context_size)
        return jnp.zeros(mem_shape, dtype=self._complex_dtype_from(self.param_dtype))

    @property
    def num_feature_axes(self) -> int:
        return 1

    def _ab_constants(self) -> tuple[Array, Array]:
        a = self.variable(
            "constants",
            "a",
            lambda: jnp.linspace(
                -jnp.e, -1e-6, self.memory_size, dtype=self.param_dtype
            ),
        ).value
        b = self.variable(
            "constants",
            "b",
            lambda: (2 * jnp.pi)
            / jnp.linspace(
                float(self.min_period),
                float(self.max_period),
                self.context_size,
                dtype=self.param_dtype,
            ),
        ).value
        return a, b

    def _log_gamma(self, a, b, t):
        a = jnp.clip(a[:, None], a_max=-jnp.array(1e-6, dtype=self.param_dtype))
        b = b[None, :]
        ab = jax.lax.complex(a, b)
        return ab * t 

    def _gamma(self, a, b, t):
        return jnp.exp(self._log_gamma(a, b, t))

    def _aggregate(self, x, state, dt: float = 1.0):
        a, b = self._ab_constants()

        x = jnp.repeat(x[..., None], self.context_size, axis=-1)
        x = jax.lax.complex(x, jnp.zeros_like(x))

        gamma = self._gamma(a, b, jnp.asarray(dt, dtype=self.param_dtype))

        state = state * gamma + x
        return state

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
        ln = LayerNorm(use_scale=False, use_bias=False, name="ln")

        gate_in = sigmoid(dense(features=self.memory_size, name="gate_in")(inputs))

        gated_x = dense(features=self.memory_size, name="pre")(inputs) * gate_in

        carry = self._aggregate(gated_x, carry)

        z_in = jnp.concatenate([jnp.real(carry), jnp.imag(carry)], axis=-1)
        z_in = z_in.reshape((*z_in.shape[:-2], -1))

        z = dense(features=self.features, name="mix")(z_in)

        gate_out = sigmoid(dense(features=self.features, name="gate_out")(inputs))
        skip = dense(features=self.features, name="skip")(inputs)
        y = ln(z * gate_out) + skip * (1.0 - gate_out)

        return carry, y
