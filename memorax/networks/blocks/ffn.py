from typing import Callable, Optional

import flax.linen as nn
import jax.numpy as jnp

from memorax.utils.typing import Array, Carry

from .base import Block


class FFN(nn.Module, Block):
    """Standard feed-forward network: Dense -> Activation -> Dense.

    Args:
        dim: Output dimension.
        expansion_factor: Hidden dimension multiplier (default 4).
        activation: Activation function (default GELU).
        dropout_rate: Dropout rate after activation (default 0.0).
        kernel_init: Kernel initializer.
        bias_init: Bias initializer.
    """

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
            name="up_proj",
        )(inputs)
        x = self.activation(x)
        x = nn.Dropout(
            rate=self.dropout_rate, deterministic=not self.has_rng("dropout")
        )(x)
        x = nn.Dense(
            self.features,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
            name="down_proj",
        )(x)

        return None, x


class GatedFFN(nn.Module, Block):
    """Gated feed-forward network (SwiGLU-style): Dense -> split -> act(gate) * value -> Dense.

    This implements the gated linear unit variant used in models like LLaMA and PaLM.

    Args:
        dim: Output dimension.
        expansion_factor: Hidden dimension multiplier (default 4).
        activation: Gate activation function (default GELU).
        dropout_rate: Dropout rate before output projection (default 0.0).
        kernel_init: Kernel initializer.
        use_bias: Whether to use bias in dense layers (default False for SwiGLU).
    """

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
