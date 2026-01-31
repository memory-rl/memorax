from typing import TYPE_CHECKING, Callable, Optional

from chex import PRNGKey

if TYPE_CHECKING:
    pass
else:
    pass

import chex
import jax
import jax.numpy as jnp
from flashbax import utils
from flashbax.buffers.trajectory_buffer import (Experience, TrajectoryBuffer,
                                                TrajectoryBufferSample,
                                                TrajectoryBufferState,
                                                make_trajectory_buffer)
from flashbax.utils import add_dim_to_args


def get_full_start_flags(experience: Experience) -> jnp.ndarray:
    return jnp.ones_like(experience.prev_done)


def get_start_flags_from_done(experience: Experience) -> jnp.ndarray:
    return jnp.roll(experience.prev_done, shift=1, axis=1)


def validate_sample_batch_size(sample_batch_size: int, max_length: int):
    if sample_batch_size > max_length:
        raise ValueError("sample_batch_size must be less than or equal to max_length")


def validate_min_length(min_length: int, add_batch_size: int, max_length: int):
    used_min_length = min_length // add_batch_size
    max_length_time_axis = max_length // add_batch_size
    if used_min_length > max_length_time_axis:
        raise ValueError("min_length used is too large for the buffer size.")


def validate_max_length_add_batch_size(max_length: int, add_batch_size: int):
    if max_length // add_batch_size < 2:
        raise ValueError(
            f"""max_length//add_batch_size must be greater than 2. It is currently
            {max_length}//{add_batch_size} = {max_length // add_batch_size}"""
        )


def validate_sample_sequence_length(
    sample_sequence_length: int, max_length: int, add_batch_size: int
):
    if sample_sequence_length > max_length // add_batch_size:
        raise ValueError(
            "sample_sequence_length must be <= max_length // add_batch_size"
        )


def validate_episode_buffer_args(
    max_length: int,
    min_length: int,
    sample_batch_size: int,
    sample_sequence_length: int,
    add_batch_size: int,
):
    validate_sample_batch_size(sample_batch_size, max_length)
    validate_min_length(min_length, add_batch_size, max_length)
    validate_max_length_add_batch_size(max_length, add_batch_size)
    validate_sample_sequence_length(sample_sequence_length, max_length, add_batch_size)


def _valid_start_mask(
    state: TrajectoryBufferState[Experience], sample_sequence_length: int
) -> jnp.ndarray:
    _, max_length_time_axis = utils.get_tree_shape_prefix(state.experience, n_axes=2)
    time_indices = jnp.arange(max_length_time_axis)

    def _not_full():
        last_valid = jnp.maximum(state.current_index - sample_sequence_length, -1)
        return (time_indices >= 0) & (time_indices <= last_valid)

    def _full():
        return jnp.ones((max_length_time_axis,), dtype=bool)

    return jax.lax.cond(state.is_full, _full, _not_full)


def make_episode_buffer(
    max_length: int,
    min_length: int,
    sample_batch_size: int,
    sample_sequence_length: int,
    get_start_flags: Callable[[Experience], jnp.ndarray] = get_start_flags_from_done,
    add_sequences: bool = False,
    add_batch_size: Optional[int] = None,
    min_length_time_axis: Optional[int] = None,
) -> TrajectoryBuffer:
    if add_batch_size is None:
        add_batch_size = 1
        add_batches = False
    else:
        add_batches = True

    validate_episode_buffer_args(
        max_length=max_length,
        min_length=min_length,
        sample_batch_size=sample_batch_size,
        sample_sequence_length=sample_sequence_length,
        add_batch_size=add_batch_size,
    )

    buffer = make_trajectory_buffer(
        max_length_time_axis=max_length // add_batch_size,
        min_length_time_axis=max(min_length // add_batch_size, sample_sequence_length),
        add_batch_size=add_batch_size,
        sample_batch_size=sample_batch_size,
        sample_sequence_length=sample_sequence_length,
        period=1,
        max_size=None,
    )

    add_fn = buffer.add
    if not add_batches:
        add_fn = add_dim_to_args(
            add_fn, axis=0, starting_arg_index=1, ending_arg_index=2
        )
    if not add_sequences:
        axis = 1 - int(not add_batches)
        add_fn = add_dim_to_args(
            add_fn, axis=axis, starting_arg_index=1, ending_arg_index=2
        )

    def sample_fn(
        state: TrajectoryBufferState[Experience], rng_key: PRNGKey
    ) -> TrajectoryBufferSample[Experience]:
        add_batch_size, max_length_time_axis = utils.get_tree_shape_prefix(
            state.experience, n_axes=2
        )

        start_flags = get_start_flags(state.experience)
        chex.assert_shape(start_flags, (add_batch_size, max_length_time_axis))
        start_flags = start_flags.astype(jnp.float32)

        valid_mask = _valid_start_mask(state, sample_sequence_length).astype(
            jnp.float32
        )
        start_mask = start_flags * valid_mask[None, :]

        per_row = jnp.sum(start_mask, axis=1)
        total = jnp.sum(per_row)

        def _sample_from_starts(key):
            p_rows = per_row / total
            key_rows, key_starts = jax.random.split(key)

            rows = jax.random.choice(
                key_rows,
                a=add_batch_size,
                shape=(sample_batch_size,),
                p=p_rows,
                replace=True,
            )

            row_start_probs = start_mask[rows]
            row_start_probs = row_start_probs / jnp.maximum(
                jnp.sum(row_start_probs, axis=1, keepdims=True), 1.0
            )
            keys = jax.random.split(key_starts, sample_batch_size)
            starts = jax.vmap(
                lambda k, p: jax.random.choice(
                    k, a=max_length_time_axis, shape=(), p=p, replace=True
                )
            )(keys, row_start_probs)
            return rows, starts

        def _fallback_beginning(key):
            rows = jax.random.randint(key, (sample_batch_size,), 0, add_batch_size)
            starts = jnp.zeros((sample_batch_size,), dtype=jnp.int32)
            return rows, starts

        rows, starts = jax.lax.cond(
            total > 0, _sample_from_starts, _fallback_beginning, rng_key
        )

        time_idx = (
            starts[:, None] + jnp.arange(sample_sequence_length)
        ) % max_length_time_axis
        experience = jax.tree.map(
            lambda x: x[rows[:, None], time_idx], state.experience
        )
        return TrajectoryBufferSample(experience=experience)

    return buffer.replace(add=add_fn, sample=sample_fn)
