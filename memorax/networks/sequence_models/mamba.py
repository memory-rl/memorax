from typing import Tuple

import jax.numpy as jnp
from flax import linen as nn
from flax.typing import Dtype
from memorax.utils.typing import Array, Carry

from .memoroid import Algebra


class Mamba(Algebra):
    """Mamba selective SSM algebra.

    Uses input-dependent dynamics (dt, B, C) for content-aware state updates.
    The dt, B, C parameters are computed externally and passed via kwargs.

    Element: (decay, state)
    Combine: (a_j * a_i, a_j * s_i + s_j)
    """

    num_heads: int
    head_dim: int
    hidden_dim: int
    dtype: Dtype = jnp.float32
    param_dtype: Dtype = jnp.float32

    def setup(self):
        self.log_decay = self.param(
            "log_decay", nn.initializers.normal(stddev=0.1), (self.num_heads,)
        )
        self.skip_weight = self.param(
            "skip_weight", nn.initializers.ones, (self.num_heads, self.head_dim)
        )

    def __call__(
        self,
        x: Array,
        dt: Array,
        B: Array,
        **kwargs,
    ) -> Carry:
        """Compute Mamba elements.

        Args:
            x: Input of shape (B, T, num_heads, head_dim)
            dt: Time deltas of shape (B, T, num_heads)
            B: Input projection of shape (B, T, num_heads, hidden_dim)
        """
        decay_rate = -jnp.exp(self.log_decay)
        decay = jnp.exp(dt * decay_rate[None, None, :])
        decay = decay[:, :, :, None, None]

        # State: outer product of B and x, scaled by dt
        state = jnp.einsum("bthn,bthd->bthnd", B * dt[:, :, :, None], x)

        return (decay, state)

    def combine(self, a: Carry, b: Carry) -> Carry:
        """Diagonal SSM combine: (a_j * a_i, a_j * s_i + s_j)"""
        decay_i, state_i = a
        decay_j, state_j = b
        return (decay_j * decay_i, decay_j * state_i + state_j)

    def read(self, h: Carry, x: Array, C: Array, **kwargs) -> Array:
        """Compute output from accumulated state.

        Args:
            h: Accumulated state (decay, state)
            x: Original input of shape (B, T, num_heads, head_dim)
            C: Output projection of shape (B, T, num_heads, hidden_dim)
        """
        _, state = h
        output = jnp.einsum("bthn,bthnd->bthd", C, state)
        output = output + self.skip_weight[None, None, :, :] * x
        return output

    def initialize_carry(self, key, input_shape: Tuple[int, ...]) -> Carry:
        batch_size, *_ = input_shape
        decay = jnp.ones(
            (batch_size, 1, self.num_heads, 1, 1),
            dtype=self.dtype,
        )
        state = jnp.zeros(
            (batch_size, 1, self.num_heads, self.hidden_dim, self.head_dim),
            dtype=self.dtype,
        )
        return (decay, state)
