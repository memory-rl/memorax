"""Neural Turing Machine recurrent cell implemented in Flax.

This module implements a Neural Turing Machine (NTM) cell following
Graves et al. (2014).  The implementation mimics the API of cells in
``flax.linen.recurrent`` so that it can be used with
:class:`flax.linen.recurrent.RNN`.
"""
from __future__ import annotations

from typing import Any, Tuple

import jax
import jax.numpy as jnp
import flax.linen as nn
from flax import struct
from flax.linen import recurrent

Array = jnp.ndarray
PRNGKey = Array


@struct.dataclass
class NTMState:
    """State of the :class:`NTMCell`.

    Attributes:
        controller_state: State of the LSTM controller ``(c, h)``.
        memory: External memory matrix ``[batch, memory_size, memory_width]``.
        read_weights: Read head weightings ``[batch, R, memory_size]``.
        write_weights: Write head weightings ``[batch, W, memory_size]``.
        read_vectors: Current read vectors ``[batch, R, memory_width]``.
    """

    controller_state: Tuple[Array, Array]
    memory: Array
    read_weights: Array
    write_weights: Array
    read_vectors: Array


class NTMCell(recurrent.RNNCellBase):
    """Neural Turing Machine recurrent cell."""

    memory_size: int
    memory_width: int
    num_read_heads: int = 1
    num_write_heads: int = 1
    controller_size: int = 128
    output_size: int = 32
    shift_range: int = 1
    dtype: Any = jnp.float32
    param_dtype: Any = jnp.float32

    @property
    def num_feature_axes(self) -> int:
        return 1

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------
    def initialize_carry(
        self, rng: PRNGKey, input_shape: Tuple[int, ...]
    ) -> NTMState:
        """Initialize the NTM state.

        Args:
            rng: PRNG key.
            input_shape: shape of the input excluding the feature dimension.
        """
        batch_dims = input_shape[:-1]
        mem_shape = batch_dims + (self.controller_size,)
        controller_state = (
            jnp.zeros(mem_shape, self.param_dtype),
            jnp.zeros(mem_shape, self.param_dtype),
        )
        memory = jnp.zeros(
            batch_dims + (self.memory_size, self.memory_width), dtype=self.dtype
        )
        read_vectors = jnp.zeros(
            batch_dims + (self.num_read_heads, self.memory_width), dtype=self.dtype
        )
        uniform = jnp.full(self.memory_size, 1.0 / self.memory_size, self.dtype)
        read_weights = jnp.broadcast_to(
            uniform, batch_dims + (self.num_read_heads, self.memory_size)
        )
        write_weights = jnp.broadcast_to(
            uniform, batch_dims + (self.num_write_heads, self.memory_size)
        )
        return NTMState(
            controller_state=controller_state,
            memory=memory,
            read_weights=read_weights,
            write_weights=write_weights,
            read_vectors=read_vectors,
        )

    # ------------------------------------------------------------------
    # Core cell computation
    # ------------------------------------------------------------------
    @nn.compact
    def __call__(self, carry: NTMState, x: Array) -> Tuple[NTMState, Array]:
        """Perform a single NTM step.

        Args:
            carry: previous :class:`NTMState`.
            x: input for the current time step ``[..., input_size]``.
        Returns:
            A tuple ``(new_carry, y)`` with the updated state and output.
        """
        batch_dims = x.shape[:-1]
        eps = 1e-6
        n_shifts = 2 * self.shift_range + 1

        controller_in = jnp.concatenate(
            [x, carry.read_vectors.reshape(batch_dims + (-1,))], axis=-1
        )
        controller = recurrent.LSTMCell(
            features=self.controller_size,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="controller",
        )
        new_controller_state, controller_out = controller(
            carry.controller_state, controller_in
        )

        # ------------------------------------------------------------------
        # Head parameter generation
        # ------------------------------------------------------------------
        read_dim = self.memory_width + 1 + 1 + n_shifts + 1
        write_dim = read_dim + 2 * self.memory_width

        read_params = nn.Dense(
            self.num_read_heads * read_dim,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="read_params",
        )(controller_out)
        write_params = nn.Dense(
            self.num_write_heads * write_dim,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="write_params",
        )(controller_out)

        read_params = read_params.reshape(batch_dims + (self.num_read_heads, read_dim))
        write_params = write_params.reshape(
            batch_dims + (self.num_write_heads, write_dim)
        )

        def split_params(params: Array):
            idx = 0
            k = params[..., idx : idx + self.memory_width]
            idx += self.memory_width
            beta = params[..., idx : idx + 1]
            idx += 1
            g = params[..., idx : idx + 1]
            idx += 1
            s = params[..., idx : idx + n_shifts]
            idx += n_shifts
            gamma = params[..., idx : idx + 1]
            idx += 1
            return k, beta, g, s, gamma, idx

        r_k, r_beta, r_g, r_s, r_gamma, _ = split_params(read_params)
        w_k, w_beta, w_g, w_s, w_gamma, idx = split_params(write_params)
        w_erase = write_params[..., idx : idx + self.memory_width]
        idx += self.memory_width
        w_add = write_params[..., idx : idx + self.memory_width]

        # Non-linearities for head parameters
        r_beta = nn.softplus(r_beta) + eps
        w_beta = nn.softplus(w_beta) + eps
        r_g = nn.sigmoid(r_g)[..., 0]
        w_g = nn.sigmoid(w_g)[..., 0]
        r_s = nn.softmax(r_s, axis=-1)
        w_s = nn.softmax(w_s, axis=-1)
        r_gamma = 1.0 + nn.softplus(r_gamma)[..., 0]
        w_gamma = 1.0 + nn.softplus(w_gamma)[..., 0]
        w_erase = nn.sigmoid(w_erase)

        # ------------------------------------------------------------------
        # Addressing and memory update
        # ------------------------------------------------------------------
        def content_weighting(memory: Array, key: Array, beta: Array) -> Array:
            mem_norm = jnp.sqrt(jnp.sum(memory ** 2, axis=-1) + eps)
            key_norm = jnp.sqrt(jnp.sum(key ** 2, axis=-1) + eps)
            similarity = jnp.einsum("...nw,...hw->...hn", memory, key)
            denom = mem_norm[..., None, :] * key_norm[..., :, None] + eps
            return nn.softmax(beta * (similarity / denom), axis=-1)

        def circular_convolution(w: Array, s: Array) -> Array:
            shifted = [jnp.roll(w, i, axis=-1) for i in range(-self.shift_range, self.shift_range + 1)]
            w_stack = jnp.stack(shifted, axis=-2)
            return jnp.sum(w_stack * s[..., :, None], axis=-2)

        def address(memory: Array, prev_w: Array, k, beta, g, s, gamma) -> Array:
            wc = content_weighting(memory, k, beta)
            wi = g[..., None] * wc + (1.0 - g[..., None]) * prev_w
            ws = circular_convolution(wi, s)
            wp = ws ** gamma[..., None]
            wp = wp / (jnp.sum(wp, axis=-1, keepdims=True) + eps)
            return wp

        w_write = address(
            carry.memory, carry.write_weights, w_k, w_beta, w_g, w_s, w_gamma
        )

        erase = jnp.prod(
            1.0 - w_write[..., :, :, None] * w_erase[..., :, None, :],
            axis=-3,
        )
        add = jnp.sum(
            w_write[..., :, :, None] * w_add[..., :, None, :], axis=-3
        )
        memory = carry.memory * erase + add

        w_read = address(memory, carry.read_weights, r_k, r_beta, r_g, r_s, r_gamma)
        read_vectors = jnp.einsum("...hn,...nw->...hw", w_read, memory)

        read_flat = read_vectors.reshape(
            batch_dims + (self.num_read_heads * self.memory_width,)
        )
        output = nn.Dense(
            self.output_size,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="output_proj",
        )(jnp.concatenate([controller_out, read_flat], axis=-1))

        new_state = NTMState(
            controller_state=new_controller_state,
            memory=memory,
            read_weights=w_read,
            write_weights=w_write,
            read_vectors=read_vectors,
        )
        return new_state, output
