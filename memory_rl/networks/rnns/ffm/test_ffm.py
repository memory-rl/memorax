import jax
import jax.numpy as jnp
import flax.linen as nn
import optax

from ffm_cell import FFMCell


def test_ffm_with_rnn():
    print("Testing FFMCell with nn.RNN...")

    batch_size = 2
    seq_length = 5
    input_size = 4
    output_size = 6
    memory_size = 8
    context_size = 6

    rnn = nn.RNN(
        FFMCell(
            input_size=input_size,
            output_size=output_size,
            memory_size=memory_size,
            context_size=context_size,
        )
    )

    x = jax.random.normal(jax.random.PRNGKey(42), (batch_size, seq_length, input_size))
    print(f"Input shape: {x.shape}")

    variables = rnn.init(jax.random.PRNGKey(42), x)
    print("RNN initialized successfully")

    y = rnn.apply(variables, x)
    print(f"Output shape: {y.shape}")

    # Check shape
    expected_shape = (batch_size, seq_length, output_size)
    assert y.shape == expected_shape, f"Expected {expected_shape}, got {y.shape}"

    assert jnp.all(jnp.isfinite(y)), "Output contains non-finite values"

    variance = jnp.var(y, axis=1)  # Variance across time
    assert jnp.all(variance > 1e-8), "Output doesn't vary over time"

    print("FFMCell with nn.RNN test passed!")


def test_gradient_flow():
    print("Testing gradient flow...")

    batch_size = 2
    seq_length = 3
    input_size = 4
    output_size = 5

    rnn = nn.RNN(
        FFMCell(
            input_size=input_size,
            output_size=output_size,
            memory_size=6,
            context_size=4,
        )
    )

    x = jax.random.normal(jax.random.PRNGKey(1), (batch_size, seq_length, input_size))
    y_true = jax.random.normal(
        jax.random.PRNGKey(42), (batch_size, seq_length, output_size)
    )

    variables = rnn.init(jax.random.PRNGKey(0), x)

    def loss_fn(params):
        y_pred = rnn.apply(params, x)
        return jnp.mean((y_pred - y_true) ** 2)

    # Compute gradients
    loss_val, grads = jax.value_and_grad(loss_fn)(variables)

    print(f"Loss: {loss_val}")

    # Check that loss is finite
    assert jnp.isfinite(loss_val), "Loss is not finite"

    # Check that all gradients are finite
    def check_gradients(grad_tree):
        flat_grads = jax.tree_util.tree_flatten(grad_tree)[0]
        all_finite = True
        any_nonzero = False
        for grad in flat_grads:
            if grad is not None:
                if not jnp.all(jnp.isfinite(grad)):
                    all_finite = False
                if jnp.any(grad != 0):
                    any_nonzero = True
        return all_finite, any_nonzero

    finite, nonzero = check_gradients(grads)
    assert finite, "Some gradients are not finite"
    assert nonzero, "All gradients are zero"

    print("Gradient flow test passed!")


def test_memory_evolution():
    print("Testing memory evolution...")

    batch_size = 2
    input_size = 3
    output_size = 4
    memory_size = 4
    context_size = 3

    cell = FFMCell(
        input_size=input_size,
        output_size=output_size,
        memory_size=memory_size,
        context_size=context_size,
    )

    key = jax.random.PRNGKey(42)

    input_shape = (batch_size, input_size)
    carry = cell.initialize_carry(key, input_shape)
    inputs = jax.random.normal(key, input_shape)
    params = cell.init(key, carry, inputs)

    print(f"Initial memory state (first element): {carry.memory_state[0, 0, 0]}")
    print(f"Initial timestep: {carry.timestep}")

    current_carry = carry
    for i in range(3):
        inputs = jax.random.normal(jax.random.fold_in(key, i), input_shape)
        current_carry, output = cell.apply(params, current_carry, inputs)
        print(
            f"Step {i}: timestep = {current_carry.timestep}, "
            f"memory[0,0,0] = {current_carry.memory_state[0, 0, 0]}"
        )

    # Check that memory changed
    assert not jnp.allclose(
        carry.memory_state, current_carry.memory_state
    ), "Memory state did not change"

    # Check that timestep incremented
    assert (
        current_carry.timestep == 3.0
    ), f"Expected timestep 3.0, got {current_carry.timestep}"

    print("Memory evolution test passed!")


def test_overfit_simple_sequence():
    batch_size = 2
    seq_length = 8
    input_size = 4
    output_size = 4

    rnn = nn.RNN(
        FFMCell(
            input_size=input_size,
            output_size=output_size,
            memory_size=16,
            context_size=8,
        )
    )

    x = jax.random.normal(jax.random.PRNGKey(1), (batch_size, seq_length, input_size))
    y_true = x + 0.1 * jax.random.normal(
        jax.random.PRNGKey(2), x.shape
    )  # Simple transformation

    variables = rnn.init(jax.random.PRNGKey(0), x)

    optimizer = optax.adam(learning_rate=0.01)
    opt_state = optimizer.init(variables)

    def loss_fn(params):
        y_pred = rnn.apply(params, x)
        return jnp.mean((y_pred - y_true) ** 2)

    @jax.jit
    def update(params, opt_state):
        loss_val, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state = optimizer.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)
        return new_params, opt_state, loss_val

    for i in range(500):
        variables, opt_state, loss_val = update(variables, opt_state)
        if i % 100 == 0:
            print(f"Step {i}, Loss: {loss_val:.6f}")

    final_loss = loss_fn(variables)
    assert (
        final_loss < 0.1
    ), f"Model did not overfit simple pattern: final loss {final_loss:.6f}"


if __name__ == "__main__":
    print("Running FFMCell tests...\n")

    success = True
    success &= test_ffm_with_rnn()
    print()
    success &= test_gradient_flow()
    print()
    success &= test_memory_evolution()
    print()
    success &= test_overfit_simple_sequence()

    if success:
        print("\nAll FFMCell tests passed!")
    else:
        print("\nSome tests failed!")
