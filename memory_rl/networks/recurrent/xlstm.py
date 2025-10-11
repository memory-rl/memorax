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

from .slstm import sLSTMBlock
from .mlstm import mLSTMBlock

A = TypeVar("A")
Carry = Any
CarryHistory = Any
Output = Any


def initalize_slstm_carry(
    rng, input_shape, *, features, conv_kernel_size, carry_init, param_dtype
):
    batch = input_shape[:-1]
    k1, k2, k3, k4, k5 = random.split(rng, 5)
    mem = batch + (features,)
    c0 = carry_init(k1, mem, param_dtype)
    n0 = carry_init(k2, mem, param_dtype)
    h0 = carry_init(k3, mem, param_dtype)
    m0 = carry_init(k4, mem, param_dtype)
    buf = batch + (max(conv_kernel_size - 1, 0), features)
    convbuf0 = carry_init(k5, buf, param_dtype)
    return (c0, n0, h0, m0, convbuf0)


def initalize_mlstm_carry(
    rng,
    input_shape,
    *,
    features,
    num_heads,
    up_proj_factor,
    conv_kernel_size,
    carry_init,
    param_dtype,
):
    batch_dims = input_shape[:-1]
    up_features = features * up_proj_factor
    head_dim = up_features // num_heads
    key_c, key_n, key_m, key_conv_buf = random.split(rng, 4)
    C = carry_init(key_c, (batch_dims + (num_heads, head_dim, head_dim)), param_dtype)
    n = carry_init(key_n, (batch_dims + (num_heads, head_dim)), param_dtype)
    m = carry_init(key_m, (batch_dims + (num_heads,)), param_dtype)
    conv_buf = carry_init(
        key_conv_buf,
        batch_dims + (up_features, max(conv_kernel_size - 1, 0)),
        param_dtype,
    )
    return (C, n, m, conv_buf)


class xLSTMCell(RNNCellBase):
    features: int
    pattern: tuple[str, ...]  # sequence of "s" / "m"
    s_kwargs: dict = field(default_factory=dict)
    m_kwargs: dict = field(default_factory=dict)

    kernel_init: Any = None
    bias_init: Any = None

    @compact
    def __call__(self, carry, inputs):
        x = inputs
        cells = []
        for i, kind in enumerate(self.pattern):
            if kind == "s":
                block = sLSTMBlock(
                    self.features, name=f"sLSTMBlock_{i}", **self.s_kwargs
                )
            elif kind == "m":
                block = mLSTMBlock(
                    self.features, name=f"mLSTMBlock_{i}", **self.m_kwargs
                )
            else:
                raise ValueError(f"Unknown kind {kind!r}")
            cell, x = block(carry[i], x)
            cells.append(cell)
        return tuple(cells), x

    @nowrap
    def initialize_carry(self, rng, input_shape):
        s_defaults = {
            "conv_kernel_size": 4,
            "carry_init": initializers.zeros_init(),
            "param_dtype": jnp.float32,
        }
        m_defaults = {
            "num_heads": 4,
            "up_proj_factor": 2,
            "conv_kernel_size": 4,
            "carry_init": initializers.zeros_init(),
            "param_dtype": jnp.float32,
        }
        s_kw = {**s_defaults, **self.s_kwargs}
        m_kw = {**m_defaults, **self.m_kwargs}

        keys = random.split(rng, len(self.pattern))
        carries = []
        for key, kind in zip(keys, self.pattern):
            if kind == "s":
                carries.append(
                    initalize_slstm_carry(
                        key,
                        input_shape,
                        features=self.features,
                        conv_kernel_size=int(s_kw["conv_kernel_size"]),
                        carry_init=s_kw["carry_init"],
                        param_dtype=s_kw["param_dtype"],
                    )
                )
            else:  # "m"
                carries.append(
                    initalize_mlstm_carry(
                        key,
                        input_shape,
                        features=self.features,
                        num_heads=int(m_kw["num_heads"]),
                        up_proj_factor=int(m_kw["up_proj_factor"]),
                        conv_kernel_size=int(m_kw["conv_kernel_size"]),
                        carry_init=m_kw["carry_init"],
                        param_dtype=m_kw["param_dtype"],
                    )
                )
        return tuple(carries)

    @property
    def num_feature_axes(self) -> int:
        return 1
