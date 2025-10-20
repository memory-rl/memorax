from dataclasses import field
from typing import (
    Tuple,
    Literal,
    Any,
    TypeVar,
)

from jax import random
import jax.numpy as jnp
from flax.linen import initializers
from flax.linen.module import compact, nowrap
from flax.linen.recurrent import RNNCellBase
from flax.typing import (
    PRNGKey,
)

from .slstm import sLSTMCell
from .mlstm import mLSTMCell

A = TypeVar("A")
Carry = Any
CarryHistory = Any
Output = Any


class xLSTMCell(RNNCellBase):
    features: int
    pattern: tuple[str, ...]  # sequence of "s" / "m"

    kernel_init: Any = None
    bias_init: Any = None

    @compact
    def __call__(self, carry, inputs):
        x = inputs
        cells = []
        for i, kind in enumerate(self.pattern):
            if kind == "s":
                block = sLSTMCell(
                    self.features,
                    name=f"sLSTMCell_{i}",
                )
            elif kind == "m":
                block = mLSTMCell(
                    self.features,
                    name=f"mLSTMCell_{i}",
                )
            else:
                raise ValueError(f"Unknown kind {kind!r}")
            cell, x = block(carry[i], x)
            cells.append(cell)
        return tuple(cells), x

    @nowrap
    def initialize_carry(self, rng, input_shape):

        keys = random.split(rng, len(self.pattern))
        carries = []
        for key, kind in zip(keys, self.pattern):
            if kind == "s":
                carries.append(
                    sLSTMCell._initialize_carry(
                        key,
                        input_shape,
                        features=self.features,
                    )
                )
            else:  # "m"
                carries.append(
                    mLSTMCell._initialize_carry(
                        key,
                        input_shape,
                        features=self.features,
                    )
                )
        return tuple(carries)

    @property
    def num_feature_axes(self) -> int:
        return 1
