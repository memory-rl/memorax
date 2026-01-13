from functools import partial
from typing import Optional

import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.linen import initializers
from flax.linen.linear import Dense, default_kernel_init
from flax.linen.module import compact, nowrap
from flax.linen.normalization import LayerNorm
from flax.typing import Array, Dtype, Initializer, PRNGKey

from memorax.utils.typing import Carry, Key

from .sequence_model import SequenceModel
from .utils import get_input_shape


class FFM(SequenceModel):

    features: int
    memory_size: int
    context_size: int

    retention_horizon: int = 1024

    min_period: int = 1
    max_period: int = 1024

    epsilon: float = 0.01
    beta: float = 0.01

    kernel_init: Initializer = default_kernel_init
    bias_init: Initializer = initializers.zeros_init()
    dtype: Dtype | None = None
    param_dtype: Dtype = jnp.float32

    def _complex_dtype_from(self, param_dtype):
        return jnp.complex64 if param_dtype == jnp.float32 else jnp.complex128

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

    def _log_gamma(self, a, b, t):
        a = jnp.clip(
            a,
            a_min=jnp.array(-self.limit, dtype=self.param_dtype),
            a_max=jnp.array(-1e-8, dtype=self.param_dtype),
        )
        a = jnp.reshape(a, (1, self.memory_size, 1))
        b = jnp.reshape(b, (1, 1, self.context_size))
        ab = jax.lax.complex(a, b)

        T, *_ = t.shape
        return ab * t.reshape(T, 1, 1)

    def _gamma(self, a, b, t):
        return jnp.exp(self._log_gamma(a, b, t))

    def _aggregate(self, x, carry, mask):

        timestep = jnp.arange(-1, x.shape[0], dtype=self.param_dtype)
        timestep = jax.lax.complex(timestep, jnp.zeros_like(timestep))

        x = jnp.repeat(x.reshape(*x.shape, 1), self.context_size, axis=-1)
        x = jax.lax.complex(x, jnp.zeros_like(x))
        x = jnp.concatenate([carry, x], axis=0)

        mask = jnp.concatenate(
            [jnp.array([0]), jnp.cumsum(mask)],
            axis=0,
            dtype=jnp.int32,
        )

        carry, *_ = jax.lax.associative_scan(
            partial(self._update, self.a, self.b),
            (x, timestep, mask),
            axis=0,
        )

        return carry[1:]

    def _update(self, a, b, lhs, rhs):
        carry, i, id_lhs = lhs
        x, j, id_rhs = rhs
        done = id_rhs != id_lhs
        done = done[..., None, None]
        carry = carry * self._gamma(a, b, j - i) + x
        carry = jnp.where(done, x, carry)
        return carry, j, id_rhs

    @nowrap
    def initialize_carry(self, key: Key, input_shape: tuple[int, ...]) -> Carry:
        batch_size, *_ = input_shape
        mem_shape = (
            batch_size,
            1,
            self.memory_size,
            self.context_size,
        )
        return jnp.zeros(mem_shape, dtype=self._complex_dtype_from(self.param_dtype))

    @compact
    def __call__(
        self,
        inputs: Array,
        mask: Array,
        initial_carry: Optional[Carry] = None,
        **kwargs,
    ) -> tuple[Array, Array]:

        if initial_carry is None:
            input_shape = get_input_shape(inputs)
            initial_carry = self.initialize_carry(jax.random.key(0), input_shape)

        dense = partial(
            Dense,
            use_bias=True,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )
        ln = LayerNorm(use_scale=False, use_bias=False, name="ln")

        pre = dense(features=self.memory_size, name="pre")(inputs)
        input_gate = nn.sigmoid(
            dense(features=self.memory_size, name="input_gate")(inputs)
        )
        output_gate = nn.sigmoid(
            dense(features=self.features, name="output_gate")(inputs)
        )
        skip = dense(features=self.features, name="skip")(inputs)

        x = pre * input_gate

        carry = jax.vmap(self._aggregate, in_axes=(0, 0, 0))(x, initial_carry, mask)

        z_in = jnp.concatenate([jnp.real(carry), jnp.imag(carry)], axis=-1)
        z_in = z_in.reshape((*z_in.shape[:-2], -1))
        z = dense(features=self.features, name="mix")(z_in)

        y = ln(z) * output_gate + skip * (1.0 - output_gate)

        return carry, y
