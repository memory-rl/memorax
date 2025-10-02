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
        mem_shape = batch_dims + (1,) + (self.memory_size, self.context_size)
        return jnp.zeros(mem_shape, dtype=self._complex_dtype_from(self.param_dtype))

    @property
    def num_feature_axes(self) -> int:
        return 1

    def _get_ab(self) -> tuple[Array, Array]:
        a = self.param(
            "a",
            lambda _: jnp.linspace(
                -jnp.e, -1e-6, self.memory_size, dtype=self.param_dtype
            ),
        )
        b = self.param(
            "b",
            lambda _: (2 * jnp.pi)
            / jnp.linspace(
                float(self.min_period),
                float(self.max_period),
                self.context_size,
                dtype=self.param_dtype,
            ),
        )
        return a, b

    def _log_gamma(self, a, b, t):
        a = jnp.clip(a[:, None], a_max=jnp.array(-1e-6, dtype=self.param_dtype))
        b = b[None, :]
        ab = jax.lax.complex(a, b)
        return ab * t.reshape(t.shape[0], 1, 1)

    def _gamma(self, a, b, t):
        return jnp.exp(self._log_gamma(a, b, t))

    def _aggregate(self, x, carry, done):

        a, b = self._get_ab()

        timestep = jnp.arange(-1, x.shape[0], dtype=jnp.float32)
        timestep = jax.lax.complex(timestep, jnp.zeros_like(timestep))

        x = jnp.repeat(x.reshape(*x.shape, 1), self.context_size, axis=-1)
        x = jax.lax.complex(x, jnp.zeros_like(x))
        x = jnp.concatenate([carry, x], axis=0)

        done = jnp.concatenate([jnp.array([False]), done], axis=0)
        done = done.reshape(done.shape[0], 1, 1)

        carry, *_ = jax.lax.associative_scan(
            partial(self._associative_update, a, b),
            (x, timestep, done),
            axis=0,
        )
        return carry[1:]

    def _associative_update(self, a, b, lhs, rhs):
        carry, i, done = lhs
        x, j, done = rhs
        carry = carry * self._gamma(a, b, j-i) * jnp.logical_not(done) + x
        i = j
        return carry, i, done


    @compact
    def apply_parallel(self, carry: Array, inputs: Array, dones: Array) -> tuple[Array, Array]:
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

        carry = jax.vmap(self._aggregate, in_axes=(1, 0, 1))(gated_x, carry, dones)

        z_in = jnp.concatenate([jnp.real(carry), jnp.imag(carry)], axis=-1)
        z_in = z_in.reshape((*z_in.shape[:-2], -1))

        z = dense(features=self.features, name="mix")(z_in)

        gate_out = sigmoid(dense(features=self.features, name="gate_out")(inputs))
        skip = dense(features=self.features, name="skip")(inputs)
        y = ln(z * gate_out) + skip * (1.0 - gate_out)

        return carry, y
