from typing import (
    Optional,
    Any,
    TypeVar,
)

from jax import random
import flax.linen as nn
from flax.linen.module import compact, nowrap
from flax.linen.recurrent import RNNCellBase

from .slstm import sLSTMLayer
from .mlstm import mLSTMLayer

A = TypeVar("A")
Carry = Any
CarryHistory = Any
Output = Any


class xLSTMBlock(nn.Module):
    layer: nn.Module
    ffn: Optional[nn.Module] = None
    kernel_init: Any = None
    bias_init: Any = None

    @compact
    def __call__(self, carry, inputs):
        skip = inputs
        x = nn.LayerNorm(use_bias=False, name="pre_ln")(inputs)
        carry, x = self.layer(carry, x)
        x = x + skip

        if self.ffn is not None:
            skip = x
            x = nn.LayerNorm(use_bias=False, name="post_ln")(x)
            x = self.ffn(x) + skip
        return carry, x

    @nowrap
    def initialize_carry(self, rng, input_shape):
        return self.layer.initialize_carry(rng, input_shape)


class xLSTMCell(RNNCellBase):
    features: int
    hidden_dim: int
    pattern: tuple[str, ...]  # sequence of "s" / "m"

    kernel_init: Any = None
    bias_init: Any = None

    @compact
    def __call__(self, carry, inputs):
        x = inputs
        cells = []
        for i, kind in enumerate(self.pattern):
            if kind == "s":
                block = xLSTMBlock(
                    layer=sLSTMLayer(
                        features=self.features,
                        hidden_dim=self.hidden_dim,
                        name=f"sLSTMCell_{i}",
                    ),
                    kernel_init=self.kernel_init,
                    bias_init=self.bias_init,
                )
            elif kind == "m":
                block = xLSTMBlock(
                    layer=mLSTMLayer(
                        features=self.features,
                        hidden_dim=self.hidden_dim,
                        name=f"mLSTMCell_{i}",
                    ),
                    kernel_init=self.kernel_init,
                    bias_init=self.bias_init,
                )
            else:
                raise ValueError(f"Unknown kind {kind!r}")
            cell, x = block(carry[i], x)
            cells.append(cell)
        x = nn.LayerNorm(use_bias=False, name="post_ln")(x)
        return tuple(cells), x

    @nowrap
    def initialize_carry(self, rng, input_shape):
        keys = random.split(rng, len(self.pattern))
        carry = []
        for key, kind in zip(keys, self.pattern):
            if kind == "s":
                carry.append(
                    sLSTMLayer._initialize_carry(
                        key,
                        input_shape,
                        hidden_dim=self.hidden_dim,
                    )
                )
            elif kind == "m":
                carry.append(
                    mLSTMLayer._initialize_carry(
                        key,
                        input_shape,
                        hidden_dim=self.hidden_dim,
                    )
                )
            else:
                raise ValueError(f"Unknown kind {kind!r}")
        return tuple(carry)
