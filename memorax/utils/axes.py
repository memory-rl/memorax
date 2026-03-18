import jax
import jax.numpy as jnp

from memorax.utils.typing import Array


def add_time_axis(x: jax.Array):
    return x[:, None, ...]


def remove_time_axis(x: jax.Array):
    return x.squeeze(1)


def head(x: jax.Array):
    return x[:, :1]


def tail(x: jax.Array):
    return x[:, 1:]


def init(x: jax.Array):
    return x[:, :-1]


def last(x: jax.Array):
    return x[:, -1:]


def add_feature_axis(x: jax.Array):
    return x[..., None]


def remove_feature_axis(x: jax.Array):
    return x.squeeze(-1)


def get_time_axis(inputs: jax.Array, num_feature_axes=1):
    time_axis = inputs.ndim - (num_feature_axes + 1)
    if time_axis < 0:
        time_axis += inputs.ndim
    return time_axis


def get_input_shape(inputs: jax.Array, num_feature_axes=1):
    time_axis = get_time_axis(inputs, num_feature_axes)
    input_shape = inputs.shape[:time_axis] + inputs.shape[time_axis + 1 :]
    return input_shape


def get_time_axis_and_input_shape(inputs: jax.Array, num_feature_axes=1):
    time_axis = get_time_axis(inputs, num_feature_axes)
    input_shape = get_input_shape(inputs, num_feature_axes)
    return time_axis, input_shape


def ensure_axis(value, size):
    value = jnp.atleast_1d(jnp.asarray(value))
    return jnp.broadcast_to(value, (size,))


def broadcast_done(done: jax.Array, carry: jax.Array) -> jax.Array:
    while done.ndim != carry.ndim:
        done = done[..., None] if done.ndim < carry.ndim else done[..., 0]
    return done


def reset_carry(done, carry, initial_carry):
    return jax.tree.map(
        lambda initial_carry, carry: jnp.where(
            broadcast_done(done, carry), initial_carry, carry
        ),
        initial_carry,
        carry,
    )
