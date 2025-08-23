import flax.linen as nn
import jax
import jax.numpy as jnp

from .ntm import NTMCell


def test_ntm_cell_shapes_and_gradients():
    key = jax.random.key(0)
    T, B, input_size = 5, 3, 10
    cell = NTMCell(
        memory_size=32,
        memory_width=16,
        num_read_heads=1,
        num_write_heads=1,
        controller_size=64,
        output_size=8,
    )
    rnn = nn.recurrent.RNN(cell, time_major=True)
    xs = jax.random.normal(key, (T, B, input_size))

    params = rnn.init(jax.random.key(1), xs)
    ys = rnn.apply(params, xs)
    assert ys.shape == (T, B, 8)

    final_carry, ys2 = rnn.apply(params, xs, return_carry=True)
    assert jnp.allclose(ys, ys2)

    rw = final_carry.read_weights
    ww = final_carry.write_weights
    assert jnp.allclose(rw.sum(axis=-1), 1.0, atol=1e-5)
    assert jnp.all(rw >= 0)
    assert jnp.allclose(ww.sum(axis=-1), 1.0, atol=1e-5)
    assert jnp.all(ww >= 0)

    def loss_fn(p):
        y = rnn.apply(p, xs)
        return jnp.sum(y ** 2)

    grads = jax.grad(loss_fn)(params)
    flat = jax.tree_util.tree_leaves(grads)
    for g in flat:
        if g is not None:
            assert jnp.all(jnp.isfinite(g))
