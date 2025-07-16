import distrax
import flax.linen as nn
from flax.linen.initializers import constant
import jax.numpy as jnp


class DiscreteQNetwork(nn.Module):
    action_dim: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        q_values = nn.Dense(self.action_dim)(x)
        return q_values


class ContinuousQNetwork(nn.Module):

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        q_values = nn.Dense(1)(x)
        return q_values


class VNetwork(nn.Module):

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        v_value = nn.Dense(1)(x)
        return v_value


class Categorical(nn.Module):
    action_dim: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> distrax.Categorical:
        logits = nn.Dense(self.action_dim)(x)

        return distrax.Categorical(logits=logits)


class SquashedGaussian(nn.Module):
    action_dim: int

    LOG_STD_MIN = -5
    LOG_STD_MAX = 2

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        mean = nn.Dense(self.action_dim)(x)

        log_std = nn.Dense(self.action_dim)(x)
        log_std = jnp.clip(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
        std = jnp.exp(log_std)

        dist = distrax.MultivariateNormalDiag(loc=mean, scale_diag=std)
        return distrax.Transformed(dist, distrax.Block(distrax.Tanh(), ndims=1))


class Temperature(nn.Module):
    initial_temperature: float

    @nn.compact
    def __call__(self) -> jnp.ndarray:
        log_temp = self.param(
            "log_temp",
            constant(jnp.log(self.initial_temperature)),
        )
        return jnp.exp(log_temp)
