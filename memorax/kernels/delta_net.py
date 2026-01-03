import jax
import jax.numpy as jnp
from jax import custom_vjp
from jax.experimental import pallas as pl
import functools


def delta_rule_fwd_kernel(
    q_ref,
    k_ref,
    v_ref,
    beta_ref,
    mask_ref,
    initial_carry_ref,
    output_ref,
    carry_ref,
    cache_ref,
):
    _, sequence_length, *_ = q_ref.shape

    carry = initial_carry_ref[0, 0]

    def body_fn(i, carry):
        q = q_ref[0, i, 0]
        k = k_ref[0, i, 0]
        v = v_ref[0, i, 0]
        beta = beta_ref[0, i, 0]
        mask = mask_ref[0, i]

        cache_ref[0, i, 0] = carry

        carry = carry * (1.0 - mask)

        delta = v - jnp.dot(carry, k)

        update = jnp.outer(delta, k)
        carry = carry + (beta * update)

        y = jnp.dot(carry, q)

        output_ref[0, i, 0] = y

        return carry

    carry = jax.lax.fori_loop(0, sequence_length, body_fn, carry)

    carry_ref[0, 0] = carry


def delta_rule_bwd_kernel(
    q_ref,
    k_ref,
    v_ref,
    beta_ref,
    mask_ref,
    initial_carry_ref,
    carry_ref,
    d_output_ref,
    d_carry_ref,
    d_q_ref,
    d_k_ref,
    d_v_ref,
    d_beta_ref,
    d_mask_ref,
    d_initial_carry_ref,
):
    _, sequence_length, *_ = q_ref.shape

    d_carry = d_carry_ref[0, 0]

    def body_fn(i, d_carry):
        t = sequence_length - 1 - i

        q = q_ref[0, t, 0]
        k = k_ref[0, t, 0]
        v = v_ref[0, t, 0]
        beta = beta_ref[0, t, 0]
        mask = mask_ref[0, t]

        carry = carry_ref[0, t, 0]

        d_output = d_output_ref[0, t, 0]

        carry = carry * (1.0 - mask)
        delta = v - jnp.dot(carry, k)
        update = jnp.outer(delta, k)
        carry = carry + (beta * update)

        d_q = jnp.dot(carry.T, d_output)
        d_q_ref[0, t, 0] = d_q

        d_carry = d_carry + jnp.outer(d_output, q)

        d_beta = jnp.dot(jnp.dot(d_carry, k), delta)
        d_beta_ref[0, t, 0] = d_beta

        d_update = d_carry * beta

        d_delta = jnp.dot(d_update, k)

        d_v = d_delta
        d_v_ref[0, t, 0] = d_v

        d_k = jnp.dot(d_update.T, delta) - jnp.dot(carry.T, d_delta)
        d_k_ref[0, t, 0] = d_k

        d_carry = d_carry - jnp.outer(d_delta, k)

        d_mask = -jnp.sum(d_carry * carry)
        d_mask_ref[0, t] = d_mask

        d_carry = d_carry * (1.0 - mask)

        return d_carry

    d_initial_carry = jax.lax.fori_loop(0, sequence_length, body_fn, d_carry)

    d_initial_carry_ref[0, 0] = d_initial_carry


def _delta_rule_fwd(q, k, v, beta, mask, initial_carry):
    B, L, H, D = q.shape
    grid = (B, H)

    in_spec_seq = pl.BlockSpec((1, L, 1, D), lambda b, h: (b, 0, h, 0))
    in_spec_scal = pl.BlockSpec((1, L, 1), lambda b, h: (b, 0, h))
    in_spec_mask = pl.BlockSpec((1, L), lambda b, h: (b, 0))
    in_spec_carry = pl.BlockSpec((1, 1, D, D), lambda b, h: (b, h, 0, 0))

    out_spec_seq = pl.BlockSpec((1, L, 1, D), lambda b, h: (b, 0, h, 0))
    out_spec_carry = pl.BlockSpec((1, 1, D, D), lambda b, h: (b, h, 0, 0))
    out_spec_cache = pl.BlockSpec((1, L, 1, D, D), lambda b, h: (b, 0, h, 0, 0))

    output, carry, cache = pl.pallas_call(
        delta_rule_fwd_kernel,
        out_shape=[
            jax.ShapeDtypeStruct((B, L, H, D), q.dtype),
            jax.ShapeDtypeStruct((B, H, D, D), q.dtype),
            jax.ShapeDtypeStruct((B, L, H, D, D), q.dtype),
        ],
        grid=grid,
        in_specs=[
            in_spec_seq,
            in_spec_seq,
            in_spec_seq,
            in_spec_scal,
            in_spec_mask,
            in_spec_carry,
        ],
        out_specs=[
            out_spec_seq,
            out_spec_carry,
            out_spec_cache,
        ],
        name="delta_fwd",
        interpret=True,
    )(q, k, v, beta, mask, initial_carry)

    return (output, carry), (q, k, v, beta, mask, initial_carry, cache)


def _delta_rule_bwd(residuals, grads):
    q, k, v, beta, mask, initial_carry, carry = residuals

    d_output, d_carry = grads

    if d_carry is None:
        d_carry = jnp.zeros_like(initial_carry)

    B, L, H, D = q.shape
    grid = (B, H)

    spec_seq = pl.BlockSpec((1, L, 1, D), lambda b, h: (b, 0, h, 0))
    spec_scal = pl.BlockSpec((1, L, 1), lambda b, h: (b, 0, h))
    spec_mask = pl.BlockSpec((1, L), lambda b, h: (b, 0))
    spec_carry = pl.BlockSpec((1, 1, D, D), lambda b, h: (b, h, 0, 0))
    spec_cache = pl.BlockSpec((1, L, 1, D, D), lambda b, h: (b, 0, h, 0, 0))

    d_q, d_k, d_v, d_beta, d_mask, d_initial_carry = pl.pallas_call(
        delta_rule_bwd_kernel,
        out_shape=[
            jax.ShapeDtypeStruct((B, L, H, D), q.dtype),
            jax.ShapeDtypeStruct((B, L, H, D), k.dtype),
            jax.ShapeDtypeStruct((B, L, H, D), v.dtype),
            jax.ShapeDtypeStruct((B, L, H), beta.dtype),
            jax.ShapeDtypeStruct((B, L), q.dtype),
            jax.ShapeDtypeStruct((B, H, D, D), initial_carry.dtype),
        ],
        grid=grid,
        in_specs=[
            spec_seq,
            spec_seq,
            spec_seq,
            spec_scal,
            spec_mask,
            spec_carry,
            spec_cache,
            spec_seq,
            spec_carry,
        ],
        out_specs=[spec_seq, spec_seq, spec_seq, spec_scal, spec_mask, spec_carry],
        name="delta_bwd",
        interpret=True,
    )(q, k, v, beta, mask, initial_carry, carry, d_output, d_carry)

    return d_q, d_k, d_v, d_beta, d_mask, d_initial_carry


@functools.partial(custom_vjp, nondiff_argnums=())
def delta_rule(q, k, v, beta, mask, initial_carry):
    (y, final_carry), _ = _delta_rule_fwd(q, k, v, beta, mask, initial_carry)
    return y, final_carry


delta_rule.defvjp(_delta_rule_fwd, _delta_rule_bwd)
