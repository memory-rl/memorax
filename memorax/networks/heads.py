from typing import Callable

import distrax
import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.linen.initializers import constant

from memorax.networks.sequence_models.utils import remove_feature_axis


class DiscreteQNetwork(nn.Module):
    action_dim: int
    kernel_init: nn.initializers.Initializer = nn.initializers.lecun_normal()
    bias_init: nn.initializers.Initializer = nn.initializers.zeros_init()

    @nn.compact
    def __call__(self, x: jnp.ndarray, **kwargs) -> tuple[jnp.ndarray, dict]:
        q_values = nn.Dense(
            self.action_dim, kernel_init=self.kernel_init, bias_init=self.bias_init
        )(x)
        return q_values, {}

    def loss(self, output: jnp.ndarray, aux: dict, targets: jnp.ndarray) -> jnp.ndarray:
        return 0.5 * jnp.square(output - targets).mean()


class ContinuousQNetwork(nn.Module):
    kernel_init: nn.initializers.Initializer = nn.initializers.lecun_normal()
    bias_init: nn.initializers.Initializer = nn.initializers.zeros_init()

    @nn.compact
    def __call__(
        self, x: jnp.ndarray, *, action: jnp.ndarray, **kwargs
    ) -> tuple[jnp.ndarray, dict]:
        q_values = nn.Dense(1, kernel_init=self.kernel_init, bias_init=self.bias_init)(
            jnp.concatenate([x, action], axis=-1)
        )
        return jnp.squeeze(q_values, -1), {}

    def loss(self, output: jnp.ndarray, aux: dict, targets: jnp.ndarray) -> jnp.ndarray:
        return 0.5 * jnp.square(output - targets).mean()


class VNetwork(nn.Module):
    kernel_init: nn.initializers.Initializer = nn.initializers.lecun_normal()
    bias_init: nn.initializers.Initializer = nn.initializers.zeros_init()

    @nn.compact
    def __call__(self, x: jnp.ndarray, **kwargs) -> tuple[jnp.ndarray, dict]:
        v_value = nn.Dense(1, kernel_init=self.kernel_init, bias_init=self.bias_init)(x)
        return v_value, {}

    def loss(self, output: jnp.ndarray, aux: dict, targets: jnp.ndarray) -> jnp.ndarray:
        return 0.5 * jnp.square(output - targets).mean()


class HLGaussVNetwork(nn.Module):
    """HL-Gauss value head with two-hot cross-entropy loss."""

    num_bins: int = 101
    v_min: float = -10.0
    v_max: float = 10.0
    kernel_init: nn.initializers.Initializer = nn.initializers.lecun_normal()
    bias_init: nn.initializers.Initializer = nn.initializers.zeros_init()

    def setup(self):
        self.bin_width = (self.v_max - self.v_min) / (self.num_bins - 1)
        self.bin_centers = jnp.linspace(self.v_min, self.v_max, self.num_bins)

    @nn.compact
    def __call__(self, x: jnp.ndarray, **kwargs) -> tuple[jnp.ndarray, dict]:
        logits = nn.Dense(
            self.num_bins, kernel_init=self.kernel_init, bias_init=self.bias_init
        )(x)
        probs = jax.nn.softmax(logits, axis=-1)
        value = jnp.sum(probs * self.bin_centers, axis=-1, keepdims=True)
        return value, {"logits": logits}

    @nn.nowrap
    def loss(self, output: jnp.ndarray, aux: dict, targets: jnp.ndarray) -> jnp.ndarray:
        """Two-hot cross-entropy loss."""
        logits = aux["logits"]
        bin_width = (self.v_max - self.v_min) / (self.num_bins - 1)

        targets = remove_feature_axis(targets)
        targets = jnp.clip(targets, self.v_min, self.v_max)

        lower_idx = ((targets - self.v_min) / bin_width).astype(jnp.int32)
        lower_idx = jnp.clip(lower_idx, 0, self.num_bins - 2)
        upper_idx = lower_idx + 1

        upper_weight = (targets - (self.v_min + lower_idx * bin_width)) / bin_width
        lower_weight = 1.0 - upper_weight

        log_probs = jax.nn.log_softmax(logits, axis=-1)

        lower_log_prob = jnp.take_along_axis(
            log_probs, lower_idx[..., None].astype(jnp.int32), axis=-1
        ).squeeze(-1)
        upper_log_prob = jnp.take_along_axis(
            log_probs, upper_idx[..., None].astype(jnp.int32), axis=-1
        ).squeeze(-1)

        loss = -(lower_weight * lower_log_prob + upper_weight * upper_log_prob)
        return loss.mean()


class C51QNetwork(nn.Module):
    action_dim: int
    num_atoms: int = 51
    v_min: float = -10.0
    v_max: float = 10.0
    kernel_init: nn.initializers.Initializer = nn.initializers.lecun_normal()
    bias_init: nn.initializers.Initializer = nn.initializers.zeros_init()

    def setup(self):
        self.delta_z = (self.v_max - self.v_min) / (self.num_atoms - 1)
        self.atoms = jnp.linspace(self.v_min, self.v_max, self.num_atoms)

    @nn.compact
    def __call__(self, x: jnp.ndarray, **kwargs) -> tuple[jnp.ndarray, dict]:
        logits = nn.Dense(
            self.action_dim * self.num_atoms,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
        )(x)

        batch_shape = logits.shape[:-1]
        logits = logits.reshape(*batch_shape, self.action_dim, self.num_atoms)

        probs = jax.nn.softmax(logits, axis=-1)
        q_values = jnp.sum(probs * self.atoms, axis=-1)

        return q_values, {"logits": logits, "probs": probs}

    @nn.nowrap
    def loss(self, output: jnp.ndarray, aux: dict, targets: jnp.ndarray) -> jnp.ndarray:
        logits = aux["logits"]
        delta_z = (self.v_max - self.v_min) / (self.num_atoms - 1)

        targets = jnp.clip(targets, self.v_min, self.v_max)

        lower_idx = ((targets - self.v_min) / delta_z).astype(jnp.int32)
        lower_idx = jnp.clip(lower_idx, 0, self.num_atoms - 2)
        upper_idx = lower_idx + 1

        upper_weight = (targets - (self.v_min + lower_idx * delta_z)) / delta_z
        lower_weight = 1.0 - upper_weight

        log_probs = jax.nn.log_softmax(logits, axis=-1)

        lower_log_prob = jnp.take_along_axis(
            log_probs, lower_idx[..., None].astype(jnp.int32), axis=-1
        ).squeeze(-1)
        upper_log_prob = jnp.take_along_axis(
            log_probs, upper_idx[..., None].astype(jnp.int32), axis=-1
        ).squeeze(-1)

        loss = -(lower_weight * lower_log_prob + upper_weight * upper_log_prob)
        return loss.mean()


class Categorical(nn.Module):
    action_dim: int
    kernel_init: nn.initializers.Initializer = nn.initializers.lecun_normal()
    bias_init: nn.initializers.Initializer = nn.initializers.zeros_init()

    @nn.compact
    def __call__(self, x: jnp.ndarray, **kwargs) -> tuple[distrax.Categorical, dict]:
        logits = nn.Dense(
            self.action_dim, kernel_init=self.kernel_init, bias_init=self.bias_init
        )(x)

        return distrax.Categorical(logits=logits), {}


class Gaussian(nn.Module):
    action_dim: int
    transform: Callable | distrax.Bijector = lambda x: x
    kernel_init: nn.initializers.Initializer = nn.initializers.lecun_normal()
    bias_init: nn.initializers.Initializer = nn.initializers.zeros_init()

    @nn.compact
    def __call__(self, x: jnp.ndarray, **kwargs) -> tuple[distrax.Transformed, dict]:
        mean = nn.Dense(
            self.action_dim, kernel_init=self.kernel_init, bias_init=self.bias_init
        )(x)

        log_std = self.param("log_std", nn.initializers.zeros, self.action_dim)
        std = jnp.exp(log_std)
        dist = distrax.MultivariateNormalDiag(loc=mean, scale_diag=std)

        bijector = self.transform
        if not isinstance(bijector, distrax.Bijector):
            bijector = distrax.Lambda(bijector)
        bijector = distrax.Block(bijector, ndims=1)

        return distrax.Transformed(dist, bijector), {}


class SquashedGaussian(nn.Module):
    action_dim: int
    kernel_init: nn.initializers.Initializer = nn.initializers.lecun_normal()
    bias_init: nn.initializers.Initializer = nn.initializers.zeros_init()

    LOG_STD_MIN = -10
    LOG_STD_MAX = 2

    @nn.compact
    def __call__(self, x: jnp.ndarray, **kwargs) -> tuple[distrax.Transformed, dict]:
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
        return distrax.Transformed(dist, distrax.Block(distrax.Tanh(), ndims=1)), {}


class Alpha(nn.Module):
    initial_alpha: float

    @nn.compact
    def __call__(self) -> jnp.ndarray:
        log_alpha = self.param(
            "log_temp",
            constant(jnp.log(self.initial_alpha)),
            (),
        )
        return log_alpha


class Beta(nn.Module):
    initial_beta: float

    @nn.compact
    def __call__(self) -> jnp.ndarray:
        log_beta = self.param(
            "log_temp",
            constant(jnp.log(self.initial_beta)),
            (),
        )
        return log_beta


class CentralizedVNetwork(nn.Module):
    """Value network head for MAPPO that outputs values for all agents."""

    num_agents: int
    kernel_init: nn.initializers.Initializer = nn.initializers.lecun_normal()
    bias_init: nn.initializers.Initializer = nn.initializers.zeros_init()

    @nn.compact
    def __call__(self, x: jnp.ndarray, **kwargs) -> tuple[jnp.ndarray, dict]:
        v_values = nn.Dense(
            self.num_agents,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
        )(x)
        return v_values, {}

    def loss(self, output: jnp.ndarray, aux: dict, targets: jnp.ndarray) -> jnp.ndarray:
        return 0.5 * jnp.square(output - targets).mean()
