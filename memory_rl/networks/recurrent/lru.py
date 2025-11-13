from functools import partial
import jax
import jax.numpy as jnp
from flax import linen as nn

from memory_rl.networks.recurrent.utils import add_time_axis


@jax.vmap
def _binary_operator_reset(q_i, q_j):
    a_i, b_i, c_i = q_i
    a_j, b_j, c_j = q_j
    a_out = (a_j * a_i) * (1.0 - c_j) + a_j * c_j
    b_out = (a_j * b_i + b_j) * (1.0 - c_j) + b_j * c_j
    c_out = c_i * (1.0 - c_j) + c_j
    return a_out, b_out, c_out


def matrix_init(key, shape, dtype=jnp.float32, normalization=1):
    return jax.random.normal(key=key, shape=shape, dtype=dtype) / normalization


def nu_init(key, shape, r_min, r_max, dtype=jnp.float32):
    u = jax.random.uniform(key=key, shape=shape, dtype=dtype)
    return jnp.log(-0.5 * jnp.log(u * (r_max**2 - r_min**2) + r_min**2))


def theta_init(key, shape, max_phase, dtype=jnp.float32):
    u = jax.random.uniform(key, shape=shape, dtype=dtype)
    return jnp.log(max_phase * u)


def gamma_log_init(key, lamb):
    nu, theta = lamb
    diag_lambda = jnp.exp(-jnp.exp(nu) + 1j * jnp.exp(theta))
    return jnp.log(jnp.sqrt(1 - jnp.abs(diag_lambda) ** 2))


class LRULayer(nn.Module):

    features: int
    hidden_dim: int
    r_min: float = 0.0
    r_max: float = 1.0
    max_phase: float = 6.28

    def setup(self):
        self.theta_log = self.param(
            "theta_log",
            partial(theta_init, max_phase=self.max_phase),
            (self.hidden_dim,),
        )
        self.nu_log = self.param(
            "nu_log",
            partial(nu_init, r_min=self.r_min, r_max=self.r_max),
            (self.hidden_dim,),
        )
        self.gamma_log = self.param(
            "gamma_log", gamma_log_init, (self.nu_log, self.theta_log)
        )

        self.B_real = self.param(
            "B_real",
            partial(matrix_init, normalization=jnp.sqrt(2 * self.features)),
            (self.hidden_dim, self.features),
        )
        self.B_imag = self.param(
            "B_imag",
            partial(matrix_init, normalization=jnp.sqrt(2 * self.features)),
            (self.hidden_dim, self.features),
        )
        self.C_real = self.param(
            "C_real",
            partial(matrix_init, normalization=jnp.sqrt(self.hidden_dim)),
            (self.features, self.hidden_dim),
        )
        self.C_imag = self.param(
            "C_imag",
            partial(matrix_init, normalization=jnp.sqrt(self.hidden_dim)),
            (self.features, self.hidden_dim),
        )
        self.D = self.param("D", matrix_init, (self.features,))

    def __call__(self, inputs, mask, carry):
        """Forward pass of a LRU: h_t+1 = lambda * h_t + B x_t+1, y_t = Re[C h_t + D x_t]"""
        diag_lambda = jnp.exp(-jnp.exp(self.nu_log) + 1j * jnp.exp(self.theta_log))
        B_norm = (self.B_real + 1j * self.B_imag) * jnp.expand_dims(
            jnp.exp(self.gamma_log), axis=-1
        )
        C = jax.lax.complex(self.C_real, self.C_imag)

        Lambda_elements = jnp.repeat(diag_lambda[None, ...], inputs.shape[0], axis=0)
        Lambda_elements = jnp.concatenate(
            [jnp.ones((1, self.hidden_dim)), Lambda_elements]
        )
        Bu_elements = jax.vmap(lambda u: B_norm @ u)(inputs)
        Bu_elements = jnp.concatenate([carry, Bu_elements])
        mask = jnp.concatenate([jnp.zeros(1), mask])
        mask = add_time_axis(mask)
        _, carry, _ = jax.lax.associative_scan(
            _binary_operator_reset, (Lambda_elements, Bu_elements, mask)
        )
        carry = carry[1:]
        outputs = jax.vmap(lambda h, x: (C @ h).real + self.D * x)(carry, inputs)

        return carry, outputs


class LRUBlock(nn.Module):

    features: int
    hidden_dim: int
    dropout_rate: float = 0.0
    prenorm: bool = False
    training: bool = True

    @nn.compact
    def __call__(self, inputs, mask, carry):
        x = inputs
        if self.prenorm:
            x = nn.LayerNorm(name="pre_norm")(x)
        carry, x = nn.vmap(
            LRULayer,
            variable_axes={"params": None},
            split_rngs={"params": False},
        )(features=self.features, hidden_dim=self.hidden_dim)(x, mask, carry)
        x = nn.gelu(x)
        x = nn.Dropout(
            self.dropout_rate, broadcast_dims=(0,), deterministic=not self.training
        )(x)
        x = nn.Dense(self.hidden_dim)(x) * jax.nn.sigmoid(nn.Dense(self.hidden_dim)(x))
        x = nn.Dropout(
            self.dropout_rate, broadcast_dims=(0,), deterministic=not self.training
        )(x)
        x = inputs + x
        if not self.prenorm:
            x = nn.LayerNorm(name="post_norm")(x)
        return carry, x


class LRU(nn.Module):
    """Encoder containing several SequenceLayer"""

    features: int
    hidden_dim: int
    num_layers: int
    dropout_rate: float = 0.0
    training: bool = True
    prenorm: bool = False

    @nn.compact
    def __call__(self, inputs, mask, initial_carry):
        x = inputs
        new_carry = []
        for carry_i in initial_carry:
            new_carry_i, x = LRUBlock(
                features=self.features,
                hidden_dim=self.hidden_dim,
                dropout_rate=self.dropout_rate,
                training=self.training,
                prenorm=self.prenorm,
            )(x, mask, carry_i)
            new_carry.append(new_carry_i)
        return tuple(new_carry), x

    def initialize_carry(self, rng: jax.Array, input_shape) -> tuple:
        batch_size, *_ = input_shape

        initial_carry = jnp.zeros(
            (
                batch_size,
                1,
                self.hidden_dim,
            ),
            dtype=jnp.complex64,
        )
        return tuple(initial_carry for _ in range(self.num_layers))
