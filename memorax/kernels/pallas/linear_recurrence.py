"""Linear recurrence kernel for parallel sequence processing.

Implements h[t] = where(mask[t], x[t], a[t] * h[t-1] + x[t])
with GPU acceleration via Pallas and CPU fallback via lax.scan.
"""

import functools

import jax
import jax.numpy as jnp


def _broadcast_mask(mask, target):
    """Broadcast (B, T) mask to match target shape (B, T, *D)."""
    while mask.ndim < target.ndim:
        mask = mask[..., jnp.newaxis]
    return mask


def _sequential_recurrence(x, a, init, mask):
    """CPU fallback: sequential linear recurrence via lax.scan."""

    def step(h, inputs):
        x_t, a_t, m_t = inputs
        m_exp = _broadcast_mask(m_t, h)
        h_new = a_t * h + x_t
        h = jnp.where(m_exp, x_t, h_new)
        return h, h

    init_state = init[:, 0]
    xs = (
        jnp.moveaxis(x, 1, 0),
        jnp.moveaxis(a, 1, 0),
        jnp.moveaxis(mask, 1, 0),
    )
    h_last, h_seq = jax.lax.scan(step, init_state, xs)
    h_seq = jnp.moveaxis(h_seq, 0, 1)
    h_last = h_last[:, jnp.newaxis]
    return h_seq, h_last


def _pallas_recurrence(x, a, init, mask):
    """GPU implementation using Pallas kernel."""
    from jax.experimental import pallas as pl

    B, T, flat_D = x.shape
    tile_size = min(128, flat_D)
    pad_D = (tile_size - flat_D % tile_size) % tile_size
    if pad_D > 0:
        x = jnp.pad(x, ((0, 0), (0, 0), (0, pad_D)))
        a = jnp.pad(a, ((0, 0), (0, 0), (0, pad_D)), constant_values=1)
        init = jnp.pad(init, ((0, 0), (0, 0), (0, pad_D)))
    padded_D = x.shape[2]
    num_tiles = padded_D // tile_size

    init_2d = init[:, 0]

    def kernel_fn(x_ref, a_ref, init_ref, mask_ref, h_ref):
        h = init_ref[:]

        def body_fn(t, h):
            x_t = x_ref[t, :]
            a_t = a_ref[t, :]
            m_t = mask_ref[t]
            h_new = a_t * h + x_t
            h_out = jnp.where(m_t, x_t, h_new)
            h_ref[t, :] = h_out
            return h_out

        jax.lax.fori_loop(0, T, body_fn, h)

    h_seq = pl.pallas_call(
        kernel_fn,
        grid=(B, num_tiles),
        in_specs=[
            pl.BlockSpec((T, tile_size), lambda b, d: (b, 0, d * tile_size)),
            pl.BlockSpec((T, tile_size), lambda b, d: (b, 0, d * tile_size)),
            pl.BlockSpec((tile_size,), lambda b, d: (b, d * tile_size)),
            pl.BlockSpec((T,), lambda b, d: (b, 0)),
        ],
        out_specs=pl.BlockSpec((T, tile_size), lambda b, d: (b, 0, d * tile_size)),
        out_shape=jax.ShapeDtypeStruct((B, T, padded_D), x.dtype),
    )(x, a, init_2d, mask)

    if pad_D > 0:
        h_seq = h_seq[:, :, :flat_D]

    h_last = h_seq[:, -1:]
    return h_seq, h_last


def _dispatch_recurrence(x, a, init, mask):
    """Auto-dispatch to Pallas (GPU) or sequential (CPU) implementation."""
    if jax.default_backend() == "gpu":
        return _pallas_recurrence(x, a, init, mask)
    return _sequential_recurrence(x, a, init, mask)


@functools.partial(jax.custom_vjp, nondiff_argnums=())
def linear_recurrence(x, a, initial_carry, mask):
    """Compute linear recurrence h[t] = where(mask[t], x[t], a[t]*h[t-1]+x[t]).

    Args:
        x: State contributions (B, T, *D)
        a: Decay factors (B, T, *D)
        initial_carry: Initial state (B, 1, *D)
        mask: Reset mask (B, T) — 1 means reset

    Returns:
        h_seq: Output sequence (B, T, *D)
        h_last: Final state (B, 1, *D)
    """
    h_seq, h_last = _dispatch_recurrence(x, a, initial_carry, mask)
    return h_seq, h_last


def _linear_recurrence_fwd(x, a, initial_carry, mask):
    h_seq, h_last = linear_recurrence(x, a, initial_carry, mask)
    return (h_seq, h_last), (h_seq, x, a, initial_carry, mask)


def _linear_recurrence_bwd(res, g):
    h_seq, x, a, initial_carry, mask = res
    dh_seq, dh_last = g

    B, T = mask.shape
    mask_exp = _broadcast_mask(mask, a)

    # --- Reverse scan to compute dh_bar ---
    # JAX complex grad convention: df = Re(grad * dz), so VJP of
    # y = a * x gives grad_x = g * a (no conjugation).
    #
    # dh_bar[T-1] = dh_seq[T-1] + dh_last[:, 0]
    # dh_bar[t] = dh_seq[t] + (1-mask[t+1]) * a[t+1] * dh_bar[t+1]
    #
    # Reformulated as forward scan on reversed arrays:
    #   x_bwd = flip(dh_seq)
    #   a_bwd[:, 0] = 1, a_bwd[:, s] = (1-mask[T-s]) * a[T-s] for s>=1
    #   init_bwd = dh_last

    x_bwd = jnp.flip(dh_seq, axis=1)

    # Build a_bwd: first element is identity (1), rest are shifted reversed
    a_tail = jnp.flip(a[:, 1:], axis=1)
    mask_tail = jnp.flip(mask[:, 1:], axis=1)
    mask_tail_exp = _broadcast_mask(mask_tail, a_tail)
    a_tail_eff = (1 - mask_tail_exp) * a_tail
    a_first = jnp.ones_like(a[:, :1])
    a_bwd = jnp.concatenate([a_first, a_tail_eff], axis=1)

    init_bwd = dh_last
    mask_bwd = jnp.zeros_like(mask)

    dh_bar_rev, _ = _dispatch_recurrence(x_bwd, a_bwd, init_bwd, mask_bwd)
    dh_bar = jnp.flip(dh_bar_rev, axis=1)

    # --- Compute gradients ---
    # dx = dh_bar
    dx = dh_bar

    # da[t] = (1 - mask[t]) * dh_bar[t] * h[t-1]
    h_shifted = jnp.concatenate([initial_carry, h_seq[:, :-1]], axis=1)
    da = (1 - mask_exp) * dh_bar * h_shifted

    # dinit: gradient flowing to initial_carry from h[0]
    dinit = (1 - mask_exp[:, :1]) * a[:, :1] * dh_bar[:, :1]

    # dmask: no gradient (integer input)
    dmask = jnp.zeros_like(mask)

    return dx, da, dinit, dmask


linear_recurrence.defvjp(_linear_recurrence_fwd, _linear_recurrence_bwd)
