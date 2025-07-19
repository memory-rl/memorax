import distrax
import flax.linen as nn
import jax.numpy as jnp
from flax.linen.initializers import constant


class DiscreteQNetwork(nn.Module):
    action_dim: int
    kernel_init: nn.initializers.Initializer = nn.initializers.lecun_normal()
    bias_init: nn.initializers.Initializer = nn.initializers.zeros_init()

    @nn.compact
    def __call__(self, x: jnp.ndarray, **kwargs) -> jnp.ndarray:
        q_values = nn.Dense(
            self.action_dim, kernel_init=self.kernel_init, bias_init=self.bias_init
        )(x)
        return q_values


class ContinuousQNetwork(nn.Module):
    kernel_init: nn.initializers.Initializer = nn.initializers.lecun_normal()
    bias_init: nn.initializers.Initializer = nn.initializers.zeros_init()

    @nn.compact
    def __call__(self, x: jnp.ndarray, **kwargs) -> jnp.ndarray:
        q_values = nn.Dense(1, kernel_init=self.kernel_init, bias_init=self.bias_init)(
            x
        )
        return jnp.squeeze(q_values, -1)


class VNetwork(nn.Module):
    kernel_init: nn.initializers.Initializer = nn.initializers.lecun_normal()
    bias_init: nn.initializers.Initializer = nn.initializers.zeros_init()

    @nn.compact
    def __call__(self, x: jnp.ndarray, **kwargs) -> jnp.ndarray:
        v_value = nn.Dense(1, kernel_init=self.kernel_init, bias_init=self.bias_init)(x)
        return v_value


class Categorical(nn.Module):
    action_dim: int
    kernel_init: nn.initializers.Initializer = nn.initializers.lecun_normal()
    bias_init: nn.initializers.Initializer = nn.initializers.zeros_init()

    @nn.compact
    def __call__(self, x: jnp.ndarray, **kwargs) -> distrax.Categorical:
        logits = nn.Dense(
            self.action_dim, kernel_init=self.kernel_init, bias_init=self.bias_init
        )(x)

        return distrax.Categorical(logits=logits)


class SquashedGaussian(nn.Module):
    action_dim: int
    kernel_init: nn.initializers.Initializer = nn.initializers.lecun_normal()
    bias_init: nn.initializers.Initializer = nn.initializers.zeros_init()

    LOG_STD_MIN = -10
    LOG_STD_MAX = 2

    @nn.compact
    def __call__(self, x: jnp.ndarray, **kwargs) -> jnp.ndarray:
        temperature = kwargs.get("temperature", 1.0)

        mean = nn.Dense(
            self.action_dim, kernel_init=self.kernel_init, bias_init=self.bias_init
        )(x)

        log_std = nn.Dense(
            self.action_dim, kernel_init=self.kernel_init, bias_init=self.bias_init
        )(x)
        log_std = jnp.clip(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
        std = jnp.exp(log_std)

        dist = distrax.MultivariateNormalDiag(loc=mean, scale_diag=std * temperature)
        return distrax.Transformed(dist, distrax.Block(distrax.Tanh(), ndims=1))


class Temperature(nn.Module):
    initial_temperature: float

    @nn.compact
    def __call__(self) -> jnp.ndarray:
        log_temp = self.param(
            "log_temp",
            init_fn=constant(jnp.log(self.initial_temperature)),
            shape=(),
        )
        return jnp.exp(log_temp)
