import jax
import jax.numpy as jnp
import flax.linen as nn
from slstm import sLSTM
from mlstm import mLSTM
import optax

def test_slstm():
    batch_size = 2
    seq_length = 10
    inp_dim = 32

    rnn = nn.RNN(sLSTM(
        inp_dim=inp_dim,
        head_dim=64,
        head_num=4,
        ker_size=4,
        p_factor=4/3,
        eps=1e-8,
        use_conv=True
    ))

    x = jnp.ones((batch_size, seq_length, inp_dim))
    variables = rnn.init(jax.random.PRNGKey(42), x)
    y = rnn.apply(variables, x)

    assert y.shape == (batch_size, seq_length, inp_dim), f"Unexpected output shape: {y.shape}"

def test_mlstm():
    batch_size = 2
    seq_length = 10
    embedding_dim = 64

    rnn = nn.RNN(mLSTM(
        embedding_dim=embedding_dim,
        head_dim=16,
        num_heads=4,
        use_bias=True
    ))

    x = jnp.ones((batch_size, seq_length, embedding_dim))
    variables = rnn.init(jax.random.PRNGKey(0), x)
    y = rnn.apply(variables, x)
    assert y.shape == (batch_size, seq_length, embedding_dim), f"Unexpected output shape: {y.shape}"

def test_slstm_overfit_dummy_data():
    batch_size = 2
    seq_length = 10
    inp_dim = 4

    rnn = nn.RNN(sLSTM(
        inp_dim=inp_dim,
        head_dim=4,
        head_num=2,
        ker_size=4,
        p_factor=4/3,
        eps=1e-8,
        use_conv=True
    ))

    x = jax.random.normal(jax.random.PRNGKey(1), (batch_size, seq_length, inp_dim))
    y_true = jax.random.normal(jax.random.PRNGKey(42), (batch_size, seq_length, inp_dim))

    variables = rnn.init(jax.random.PRNGKey(0), x)

    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),  # Gradient clipping
        optax.adam(learning_rate=0.01) 
    )
    opt_state = optimizer.init(variables)

    def loss_fn(params):
        y_pred = rnn.apply(params, x)
        return jnp.mean((y_pred - y_true) ** 2)

    @jax.jit
    def update(params, opt_state):
        grads = jax.grad(loss_fn)(params)
        updates, opt_state = optimizer.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)
        return new_params, opt_state

    for _ in range(1000):
        variables, opt_state = update(variables, opt_state)

    final_loss = loss_fn(variables)
    assert final_loss < 1e-3, f"Model did not overfit: final loss {final_loss}"

def test_mlstm_overfit_dummy_data():
    batch_size = 2
    seq_length = 10
    embedding_dim = 10

    rnn = nn.RNN(mLSTM(
        embedding_dim=embedding_dim,
        head_dim=16,
        num_heads=2,
        use_bias=True
    ))

    x = jax.random.normal(jax.random.PRNGKey(1), (batch_size, seq_length, embedding_dim))
    y_true = jax.random.normal(jax.random.PRNGKey(42), (batch_size, seq_length, embedding_dim))

    variables = rnn.init(jax.random.PRNGKey(0), x)

    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),  # Gradient clipping
        optax.adam(learning_rate=4e-3) 
    )
    opt_state = optimizer.init(variables)

    def loss_fn(params):
        y_pred = rnn.apply(params, x)
        return jnp.mean((y_pred - y_true) ** 2)

    @jax.jit
    def update(params, opt_state):
        grads = jax.grad(loss_fn)(params)
        updates, opt_state = optimizer.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)
        return new_params, opt_state

    for _ in range(2500):
        variables, opt_state = update(variables, opt_state)

    final_loss = loss_fn(variables)
    assert final_loss < 1e-3, f"Model did not overfit: final loss {final_loss}"
