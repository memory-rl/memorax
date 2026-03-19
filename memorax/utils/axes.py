import jax
import jax.numpy as jnp

from memorax.utils.typing import Array, Carry


def add_time_axis(x: Array) -> Array:
    return x[:, None, ...]


def remove_time_axis(x: Array) -> Array:
    return x.squeeze(1)


def head(x: Array) -> Array:
    return x[:, :1]


def tail(x: Array) -> Array:
    return x[:, 1:]


def init(x: Array) -> Array:
    return x[:, :-1]


def last(x: Array) -> Array:
    return x[:, -1:]


def add_feature_axis(x: Array) -> Array:
    return x[..., None]


def remove_feature_axis(x: Array) -> Array:
    return x.squeeze(-1)


def get_time_axis(inputs: Array, num_feature_axes: int = 1) -> int:
    time_axis = inputs.ndim - (num_feature_axes + 1)
    if time_axis < 0:
        time_axis += inputs.ndim
    return time_axis


def get_input_shape(inputs: Array, num_feature_axes: int = 1) -> tuple:
    time_axis = get_time_axis(inputs, num_feature_axes)
    input_shape = inputs.shape[:time_axis] + inputs.shape[time_axis + 1 :]
    return input_shape


def get_time_axis_and_input_shape(inputs: Array, num_feature_axes: int = 1) -> tuple[int, tuple]:
    time_axis = get_time_axis(inputs, num_feature_axes)
    input_shape = get_input_shape(inputs, num_feature_axes)
    return time_axis, input_shape


def ensure_axis(value: Array, size: int) -> Array:
    value = jnp.atleast_1d(jnp.asarray(value))
    _, *shape = value.shape
    return jnp.broadcast_to(value, (size, *shape))


def broadcast_done(done: Array, carry: Array) -> Array:
    while done.ndim != carry.ndim:
        done = done[..., None] if done.ndim < carry.ndim else done[..., 0]
    return done


def reset_carry(done: Array, carry: Carry, initial_carry: Carry) -> Carry:
    return jax.tree.map(
        lambda initial_carry, carry: jnp.where(
            broadcast_done(done, carry), initial_carry, carry
        ),
        initial_carry,
        carry,
    )
