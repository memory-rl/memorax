import jax
import jax.numpy as jnp

from memorax.kernels.pallas import linear_recurrence as _raw_linear_recurrence
from memorax.networks.sequence_models.utils import broadcast_mask


def _accumulate_decay(decay, init_decay, mask):
    """Compute cumulative product of decay with resets via lax.scan."""
    mask_exp = mask
    while mask_exp.ndim < decay.ndim:
        mask_exp = mask_exp[..., jnp.newaxis]

    def step(h, inputs):
        d, m = inputs
        h_new = d * h
        h = jnp.where(m, d, h_new)
        return h, h

    init = init_decay[:, 0]
    xs = (jnp.moveaxis(decay, 1, 0), jnp.moveaxis(mask_exp, 1, 0))
    h_last, h_seq = jax.lax.scan(step, init, xs)
    h_seq = jnp.moveaxis(h_seq, 0, 1)
    h_last = h_last[:, jnp.newaxis]
    return h_seq, h_last


def linear_recurrence(cell, z, initial_carry, mask):
    """Scan function for Memoroid using linear recurrence kernel.

    Drop-in replacement for associative_scan for cells with binary operator:
        (decay_j * decay_i, decay_j * state_i + state_j)

    Supports LRU, S5, and Mamba cells.

    Args:
        cell: MemoroidCellBase instance (unused, kept for interface compat)
        z: Output of cell(inputs), tuple of (decay, state)
        initial_carry: Tuple of (init_decay, init_state)
        mask: Reset mask (B, T) — 1 means reset

    Returns:
        h: Carry sequence tuple (decay_seq, state_seq), each (B, T, *D)
        next_carry: Last carry tuple (decay_last, state_last), each (B, 1, *D)
    """
    decay, state = z
    init_decay, init_state = initial_carry

    # Broadcast decay to match state shape
    decay_bc = jnp.broadcast_to(decay, state.shape)

    # State recurrence: h[t] = where(mask[t], state[t], decay[t]*h[t-1]+state[t])
    h_state_seq, h_state_last = _raw_linear_recurrence(
        x=state, a=decay_bc, initial_carry=init_state, mask=mask
    )

    # Decay accumulation for carry tracking
    h_decay_seq, h_decay_last = _accumulate_decay(decay, init_decay, mask)

    h = (h_decay_seq, h_state_seq)
    next_carry = (h_decay_last, h_state_last)
    return h, next_carry
