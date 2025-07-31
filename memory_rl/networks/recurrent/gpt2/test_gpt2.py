import pytest
import jax
import jax.numpy as jnp
import numpy as np
import flax.linen as nn
import optax

from memory_rl.networks.recurrent.gpt2.gpt2 import GPTRNNCell, GPTConfig

@pytest.fixture
def small_gpt_config():
    return GPTConfig(
        block_size=6,
        vocab_size=3,
        num_layers=1,
        num_heads=2,
        num_embeds=8,
        dropout_rate=0.0,
        use_bias=True,
        dtype="float32"
    )

def test_initialize_carry_shapes_and_dtypes(small_gpt_config):
    batch_size = 2
    features = small_gpt_config.num_embeds
    cell = GPTRNNCell(max_sequence_length=small_gpt_config.block_size, config=small_gpt_config)
    carry = cell.initialize_carry(jax.random.PRNGKey(0), (batch_size, features))
    # Check kv_cache_k/v shapes: (num_blocks, batch, max_seq, num_heads, head_dim)
    num_blocks = small_gpt_config.num_layers
    num_heads = small_gpt_config.num_heads
    head_dim = small_gpt_config.num_embeds // small_gpt_config.num_heads
    assert carry.kv_cache_k.shape == (num_blocks, batch_size, small_gpt_config.block_size, num_heads, head_dim)
    assert carry.kv_cache_v.shape == (num_blocks, batch_size, small_gpt_config.block_size, num_heads, head_dim)
    assert carry.kv_cache_k.dtype == jnp.float32
    assert carry.kv_cache_v.dtype == jnp.float32
    assert carry.pos.shape == (batch_size,)
    assert carry.pos.dtype == jnp.int32

def test_gptrnncell_single_call(small_gpt_config):
    batch_size = 2
    features = small_gpt_config.num_embeds
    cell = GPTRNNCell(max_sequence_length=small_gpt_config.block_size, config=small_gpt_config)
    carry = cell.initialize_carry(jax.random.PRNGKey(1), (batch_size, features))
    x = jnp.ones((batch_size, features), dtype=jnp.float32)
    variables = cell.init(jax.random.PRNGKey(2), carry, x)
    new_carry, y = cell.apply(variables, carry, x)
    # Output shape should be (batch, num_embeds)
    assert y.shape == (batch_size, features)
    # Carry should have pos incremented by 1
    assert (new_carry.pos == carry.pos + 1).all()

def test_gptrnncell_rnn_sequence(small_gpt_config):
    batch_size = 2
    seq_len = small_gpt_config.block_size
    features = small_gpt_config.num_embeds
    cell = GPTRNNCell(max_sequence_length=seq_len, config=small_gpt_config)
    rnn = nn.RNN(cell)
    x = jnp.ones((batch_size, seq_len, features), dtype=jnp.float32)
    # RNN expects (batch, time, features)
    variables = rnn.init(jax.random.PRNGKey(3), x)
    y = rnn.apply(variables, x)
    print("RNN output shape:", y.shape)
    # Output shape: (time, features) or (batch, time, features)
    assert y.shape == (seq_len, features) or y.shape == (batch_size, seq_len, features)

def test_gptrnncell_convergence():
    config = GPTConfig(
        block_size=6,
        vocab_size=3,
        num_layers=1,
        num_heads=2,
        num_embeds=8,
        dropout_rate=0.0,
        use_bias=True,
        dtype="float32"
    )
    seq_len = config.block_size
    batch_size = 4
    cell = GPTRNNCell(max_sequence_length=seq_len, config=config)
    rnn = nn.RNN(cell)

    # simple repeating sequence: [0,1,2,0,1,2,...]
    pattern = np.arange(seq_len) % config.vocab_size
    inputs = np.tile(pattern, (batch_size, 1))
    targets = np.roll(inputs, -1, axis=1)  # Next-token prediction

    # embed layer to map tokens to input vectors
    class EmbedOnly(nn.Module):
        vocab_size: int
        features: int
        @nn.compact
        def __call__(self, x):
            return nn.Embed(self.vocab_size, self.features)(x)

    embedder = EmbedOnly(config.vocab_size, config.num_embeds)

    rng = jax.random.PRNGKey(42)
    x_tokens = jnp.array(inputs, dtype=jnp.int32)
    y_tokens = jnp.array(targets, dtype=jnp.int32)

    embed_params = embedder.init(rng, x_tokens)
    rnn_vars = rnn.init(rng, embedder.apply(embed_params, x_tokens))
    params = rnn_vars["params"]
    dense_vars = nn.Dense(config.vocab_size).init(rng, jnp.ones((batch_size, seq_len, config.num_embeds)))
    dense_params = dense_vars["params"]
    # Combine all params
    all_params = {"rnn": params, "dense": dense_params}
    all_embed_params = embed_params

    optimizer = optax.adam(learning_rate=0.05)
    opt_state = optimizer.init((all_params, all_embed_params))

    @jax.jit
    def train_step(params, embed_params, opt_state, x_tokens, y_tokens, rng):
        def total_loss(p, e):
            # Dense layer for vocab projection
            x_embed = embedder.apply(e, x_tokens)
            variables = {"params": p["rnn"]}
            result = rnn.apply(variables, x_embed, rngs={"dropout": rng})
            y_pred = result
            logits = nn.Dense(config.vocab_size).apply({"params": p["dense"]}, y_pred)
            logits = logits.reshape(-1, config.vocab_size)
            y_flat = y_tokens.reshape(-1)
            loss = optax.softmax_cross_entropy_with_integer_labels(logits, y_flat).mean()
            return loss
        grad_fn = jax.value_and_grad(total_loss, argnums=(0,1))
        (loss), grads = grad_fn(params, embed_params)
        updates, new_opt_state = optimizer.update(grads, opt_state, (params, embed_params))
        new_params, new_embed_params = optax.apply_updates((params, embed_params), updates)
        return loss, new_params, new_embed_params, new_opt_state

    losses = []
    for step in range(120):
        rng, step_rng = jax.random.split(rng)
        loss, all_params, all_embed_params, opt_state = train_step(all_params, all_embed_params, opt_state, x_tokens, y_tokens, step_rng)
        losses.append(float(loss))

    # Assert loss decreased significantly
    assert losses[0] > 0.5, f"Initial loss too low: {losses[0]}"
    assert losses[-1] < 0.2, f"Final loss not low enough: {losses[-1]}"
    assert losses[-1] < losses[0] * 0.5, f"Loss did not decrease enough: {losses[0]} -> {losses[-1]}"