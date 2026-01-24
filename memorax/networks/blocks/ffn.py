from typing import Callable, Optional

import flax.linen as nn
import jax.numpy as jnp

from memorax.utils.typing import Array, Carry

from .base import Block


class FFN(nn.Module, Block):
    """Standard feed-forward network: Dense -> Activation -> Dense."""

    features: int
    expansion_factor: int = 4
    activation: Callable = nn.gelu
    dropout_rate: float = 0.0
    kernel_init: nn.initializers.Initializer = nn.initializers.lecun_normal()
    bias_init: nn.initializers.Initializer = nn.initializers.zeros_init()

    @nn.compact
    def __call__(
        self,
        inputs: Array,
        mask: Optional[Array] = None,
        initial_carry: Optional[Carry] = None,
        **kwargs,
    ) -> tuple[Carry, Array]:
        hidden_dim = self.features * self.expansion_factor

        x = nn.Dense(
            hidden_dim,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
        )(inputs)
        x = self.activation(x)
        x = nn.Dropout(
            rate=self.dropout_rate, deterministic=not self.has_rng("dropout")
        )(x)
        x = nn.Dense(
            self.features,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
        )(x)

        return None, x

    @nn.nowrap
    def initialize_carry(self, key, input_shape):
        return self.module.initialize_carry(key, input_shape)


class GatedFFN(nn.Module, Block):
    """Gated feed-forward network (SwiGLU-style): Dense -> split -> act(gate) * value -> Dense."""

    features: int
    expansion_factor: int = 4
    activation: Callable = nn.gelu
    dropout_rate: float = 0.0
    kernel_init: nn.initializers.Initializer = nn.initializers.lecun_normal()
    use_bias: bool = False

    @nn.compact
    def __call__(
        self,
        inputs: Array,
        mask: Optional[Array] = None,
        initial_carry: Optional[Carry] = None,
        **kwargs,
    ) -> tuple[Carry, Array]:
        hidden_dim = self.features * self.expansion_factor

        x = nn.Dense(
            2 * hidden_dim,
            kernel_init=self.kernel_init,
            use_bias=self.use_bias,
            name="up_proj",
        )(inputs)
        gate, value = jnp.split(x, 2, axis=-1)
        x = self.activation(gate) * value
        x = nn.Dropout(
            rate=self.dropout_rate, deterministic=not self.has_rng("dropout")
        )(x)
        x = nn.Dense(
            self.features,
            kernel_init=self.kernel_init,
            use_bias=self.use_bias,
            name="down_proj",
        )(x)

        return None, x

    @nn.nowrap
    def initialize_carry(self, key, input_shape):
        return self.module.initialize_carry(key, input_shape)
