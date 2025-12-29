from typing import (
    Optional,
    Any,
    TypeVar,
)

from jax import random
import jax.numpy as jnp
import flax.linen as nn
from flax.linen.module import compact, nowrap
from flax.linen.recurrent import RNNCellBase

from memorax.networks.sequence_models.sequence_model import SequenceModel
from .slstm import sLSTMLayer
from .mlstm import mLSTMLayer
from .utils import small_init, wang_init

A = TypeVar("A")
Carry = Any
CarryHistory = Any
Output = Any

class GatedFeedForward(nn.Module):
    up_proj_dim: int
    down_proj_dim: int
    dropout_rate: float = 0.0

    @compact
    def __call__(self, x):
        *_, in_features = x.shape
        x = nn.Dense(2 * self.up_proj_dim, name="up_proj", kernel_init=small_init(in_features), use_bias=False)(x)
        gate, x = jnp.split(x, 2, axis=-1)
        x = nn.Dense(self.down_proj_dim, name="down_proj", kernel_init=wang_init(dim=in_features, num_blocks=1), use_bias=False)(
            nn.gelu(gate) * x
        )
        x = nn.Dropout(rate=self.dropout_rate, deterministic=self.has_rng("dropout"))(x)
        return x




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
            x = nn.LayerNorm(use_bias=False, name="ffn_ln")(x)
            x = self.ffn(x) + skip
        return carry, x

    @nowrap
    def initialize_carry(self, rng, input_shape):
        return self.layer.initialize_carry(rng, input_shape)


class xLSTMCell(RNNCellBase):
    features: int
    hidden_dim: tuple[int, ...]
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
                        hidden_dim=self.hidden_dim[i],
                        name=f"sLSTMCell_{i}",
                    ),
                    ffn=GatedFeedForward(
                        up_proj_dim=int(4/3 * self.hidden_dim[i]),
                        down_proj_dim=self.hidden_dim[i],
                    ),
                    kernel_init=self.kernel_init,
                    bias_init=self.bias_init,
                )
            elif kind == "m":
                block = xLSTMBlock(
                    layer=mLSTMLayer(
                        features=self.features,
                        hidden_dim=self.hidden_dim[i],
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
        for i, (key, kind) in enumerate(zip(keys, self.pattern)):
            if kind == "s":
                carry.append(
                    sLSTMLayer._initialize_carry(
                        key,
                        input_shape,
                        hidden_dim=self.hidden_dim[i],
                    )
                )
            elif kind == "m":
                carry.append(
                    mLSTMLayer._initialize_carry(
                        key,
                        input_shape,
                        hidden_dim=self.hidden_dim[i],
                    )
                )
            else:
                raise ValueError(f"Unknown kind {kind!r}")
        return tuple(carry)
