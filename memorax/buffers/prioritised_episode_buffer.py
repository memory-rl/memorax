"""Prioritised episode buffer combining episode-aware sampling with priority-based replay.

This buffer extends the episode buffer with prioritized experience replay (PER) as described
in https://arxiv.org/abs/1511.05952. It combines:
1. Episode-aware sampling: Only samples from valid episode start positions
2. Priority-weighted sampling: Samples proportionally to TD-error priorities
3. Importance sampling weights: For correcting the bias introduced by non-uniform sampling
"""

import functools
from typing import TYPE_CHECKING, Callable, Generic, Optional

if TYPE_CHECKING:
    from dataclasses import dataclass
else:
    from chex import dataclass

import chex
import jax
import jax.numpy as jnp
from chex import PRNGKey
from flashbax import utils
from flashbax.buffers import sum_tree
from flashbax.buffers.prioritised_trajectory_buffer import (
    SET_BATCH_FN,
    PrioritisedTrajectoryBuffer,
    PrioritisedTrajectoryBufferSample,
    PrioritisedTrajectoryBufferState,
    Probabilities,
    prioritised_init,
    set_priorities,
    validate_device,
    validate_priority_exponent,
)
from flashbax.buffers.sum_tree import SumTreeState
from flashbax.buffers.trajectory_buffer import Experience, can_sample
from flashbax.utils import add_dim_to_args
from jax import Array

from .episode_buffer import get_start_flags_from_done, validate_episode_buffer_args

Indices = Array


@dataclass(frozen=True)
class PrioritisedEpisodeBufferSample(
    PrioritisedTrajectoryBufferSample, Generic[Experience]
):
    """Sample from prioritised episode buffer with priority information.

    Attributes:
        experience: The sampled experience trajectories.
        indices: Indices corresponding to the sampled sequences (for priority updates).
        probabilities: Sampling probabilities of the sampled sequences (for importance weights).
    """

    pass


def compute_importance_weights(
    probabilities: Probabilities,
    buffer_size: int,
    beta: float = 0.4,
) -> Array:
    """Compute importance sampling weights for prioritized experience replay.

    The importance sampling weights correct for the bias introduced by non-uniform
    sampling. Weights are normalized by the maximum weight for stability.

    w_i = (N * P(i))^(-beta) / max_j(w_j)

    Args:
        probabilities: Sampling probabilities from the buffer sample.
        buffer_size: Current number of valid items in the buffer.
        beta: Importance sampling exponent. Should be annealed from initial value
            (e.g., 0.4) to 1.0 over the course of training.

    Returns:
        Normalized importance sampling weights with the same shape as probabilities.
    """
    safe_probs = jnp.maximum(probabilities, 1e-10)
    weights = (buffer_size * safe_probs) ** (-beta)
    weights = weights / jnp.maximum(weights.max(), 1e-10)
    return weights


def _valid_start_mask(
    state: PrioritisedTrajectoryBufferState[Experience], sample_sequence_length: int
) -> jnp.ndarray:
    """Get mask of valid start positions based on buffer fill state.

    Args:
        state: The buffer state.
        sample_sequence_length: Length of sequences to sample.

    Returns:
        Boolean mask of shape [max_length_time_axis] indicating valid start positions.
    """
    _, max_length_time_axis = utils.get_tree_shape_prefix(state.experience, n_axes=2)
    time_indices = jnp.arange(max_length_time_axis)

    def _not_full():
        last_valid = jnp.maximum(state.current_index - sample_sequence_length, -1)
        return (time_indices >= 0) & (time_indices <= last_valid)

    def _full():
        return jnp.ones((max_length_time_axis,), dtype=bool)

    return jax.lax.cond(state.is_full, _full, _not_full)


def _get_priorities_for_positions(
    sum_tree_state: SumTreeState,
    add_batch_size: int,
    max_length_time_axis: int,
) -> Array:
    """Get priorities for all positions in the buffer.

    Since we use period=1, each timestep position maps to one item in the sum tree.

    Args:
        sum_tree_state: The sum tree state containing priorities.
        add_batch_size: Number of parallel environments (rows in buffer).
        max_length_time_axis: Length of time axis in buffer.

    Returns:
        Array of shape [add_batch_size, max_length_time_axis] with priorities.
    """
    num_items_per_row = max_length_time_axis
    total_items = add_batch_size * num_items_per_row

    item_indices = jnp.arange(total_items)

    priorities = sum_tree.get(sum_tree_state, item_indices)

    priorities = priorities.reshape(add_batch_size, max_length_time_axis)

    return priorities


def prioritised_episode_sample(
    state: PrioritisedTrajectoryBufferState[Experience],
    rng_key: PRNGKey,
    sample_batch_size: int,
    sample_sequence_length: int,
    get_start_flags: Callable[[Experience], jnp.ndarray],
) -> PrioritisedEpisodeBufferSample[Experience]:
    """Sample episodes weighted by priority, respecting episode boundaries.

    This combines episode-aware sampling with priority-weighted sampling:
    1. Identify valid episode start positions using start_flags
    2. Mask priorities to zero for non-start positions
    3. Sample proportionally to masked priorities
    4. Return samples with indices and probabilities for importance sampling

    Args:
        state: The prioritised buffer state.
        rng_key: Random key for sampling.
        sample_batch_size: Number of sequences to sample.
        sample_sequence_length: Length of each sampled sequence.
        get_start_flags: Function to extract episode start flags from experience.

    Returns:
        PrioritisedEpisodeBufferSample with experience, indices, and probabilities.
    """
    add_batch_size, max_length_time_axis = utils.get_tree_shape_prefix(
        state.experience, n_axes=2
    )

    start_flags = get_start_flags(state.experience)
    chex.assert_shape(start_flags, (add_batch_size, max_length_time_axis))
    start_flags = start_flags.astype(jnp.float32)

    valid_mask = _valid_start_mask(state, sample_sequence_length).astype(jnp.float32)

    combined_mask = start_flags * valid_mask[None, :]

    priorities = _get_priorities_for_positions(
        state.sum_tree_state, add_batch_size, max_length_time_axis
    )

    masked_priorities = priorities * combined_mask

    flat_priorities = masked_priorities.flatten()
    total_priority = jnp.sum(flat_priorities)

    def _sample_with_priorities(key):
        """Sample using priority-weighted selection."""
        probs = flat_priorities / jnp.maximum(total_priority, 1e-10)

        flat_indices = jax.random.choice(
            key,
            a=add_batch_size * max_length_time_axis,
            shape=(sample_batch_size,),
            p=probs,
            replace=True,
        )

        rows = flat_indices // max_length_time_axis
        starts = flat_indices % max_length_time_axis

        selected_probs = probs[flat_indices]

        return rows, starts, flat_indices, selected_probs

    def _fallback_uniform(key):
        """Fallback to uniform sampling if no valid starts with priority."""
        rows = jax.random.randint(key, (sample_batch_size,), 0, add_batch_size)
        starts = jnp.zeros((sample_batch_size,), dtype=jnp.int32)
        flat_indices = rows * max_length_time_axis + starts
        uniform_prob = 1.0 / (add_batch_size * max_length_time_axis)
        selected_probs = jnp.full((sample_batch_size,), uniform_prob)
        return rows, starts, flat_indices, selected_probs

    rows, starts, flat_indices, selected_probs = jax.lax.cond(
        total_priority > 0, _sample_with_priorities, _fallback_uniform, rng_key
    )

    time_idx = (
        starts[:, None] + jnp.arange(sample_sequence_length)
    ) % max_length_time_axis

    experience = jax.tree.map(lambda x: x[rows[:, None], time_idx], state.experience)

    item_indices = flat_indices

    return PrioritisedEpisodeBufferSample(
        experience=experience,
        indices=item_indices,
        probabilities=selected_probs,
    )


def prioritised_episode_add(
    state: PrioritisedTrajectoryBufferState[Experience],
    batch: Experience,
    sample_sequence_length: int,
    device: str,
) -> PrioritisedTrajectoryBufferState[Experience]:
    """Add experience to the prioritised episode buffer.

    New items are assigned the maximum recorded priority. Items that become
    invalid (overwritten or broken by the circular buffer) have their priority
    set to zero.

    Args:
        state: Current buffer state.
        batch: Batch of experience to add with shape [add_batch_size, seq_len, ...].
        sample_sequence_length: Length of sequences that will be sampled.
        device: Device type for optimized operations ("cpu", "gpu", or "tpu").

    Returns:
        Updated buffer state with new experience and priorities.
    """
    chex.assert_tree_shape_prefix(batch, utils.get_tree_shape_prefix(state.experience))
    chex.assert_trees_all_equal_dtypes(batch, state.experience)

    add_sequence_length = utils.get_tree_shape_prefix(batch, n_axes=2)[1]
    add_batch_size, max_length_time_axis = utils.get_tree_shape_prefix(
        state.experience, n_axes=2
    )

    data_indices = (
        jnp.arange(add_sequence_length) + state.current_index
    ) % max_length_time_axis

    new_experience = jax.tree.map(
        lambda exp_field, batch_field: exp_field.at[:, data_indices].set(batch_field),
        state.experience,
        batch,
    )

    new_item_time_indices = (
        jnp.arange(add_sequence_length) + state.current_index
    ) % max_length_time_axis

    row_offsets = jnp.arange(add_batch_size)[:, None] * max_length_time_axis
    newly_valid_items = (new_item_time_indices[None, :] + row_offsets).flatten()

    new_priorities = jnp.full(
        newly_valid_items.shape, state.sum_tree_state.max_recorded_priority
    )

    invalid_time_start = (
        state.current_index - (sample_sequence_length - 1) + max_length_time_axis
    ) % max_length_time_axis

    invalid_time_indices = (
        invalid_time_start + jnp.arange(sample_sequence_length - 1)
    ) % max_length_time_axis

    newly_invalid_items = (invalid_time_indices[None, :] + row_offsets).flatten()
    invalid_priorities = jnp.zeros(newly_invalid_items.shape)

    new_sum_tree_state = SET_BATCH_FN[device](
        state.sum_tree_state,
        newly_invalid_items,
        invalid_priorities,
    )
    new_sum_tree_state = SET_BATCH_FN[device](
        new_sum_tree_state,
        newly_valid_items,
        new_priorities,
    )

    new_current_index = state.current_index + add_sequence_length
    new_running_index = state.running_index + add_sequence_length
    new_is_full = state.is_full | (new_current_index >= max_length_time_axis)
    new_current_index = new_current_index % max_length_time_axis

    return state.replace(
        experience=new_experience,
        current_index=new_current_index,
        is_full=new_is_full,
        running_index=new_running_index,
        sum_tree_state=new_sum_tree_state,
    )


def make_prioritised_episode_buffer(
    max_length: int,
    min_length: int,
    sample_batch_size: int,
    sample_sequence_length: int,
    get_start_flags: Callable[[Experience], jnp.ndarray] = get_start_flags_from_done,
    add_sequences: bool = False,
    add_batch_size: Optional[int] = None,
    priority_exponent: float = 0.6,
    device: str = "cpu",
) -> PrioritisedTrajectoryBuffer:
    """Create a prioritised episode buffer.

    This buffer combines episode-aware sampling with prioritized experience replay:
    - Only samples from valid episode start positions (identified by get_start_flags)
    - Weights sampling by TD-error priorities
    - Returns indices and probabilities for importance sampling weight computation

    Args:
        max_length: Maximum total capacity of the buffer in timesteps.
        min_length: Minimum number of timesteps before sampling is allowed.
        sample_batch_size: Number of sequences to sample per batch.
        sample_sequence_length: Length of each sampled sequence.
        get_start_flags: Function that takes experience and returns boolean array
            of shape [batch, time] indicating episode start positions.
            Defaults to get_start_flags_from_done which uses prev_done.
        add_sequences: If True, expect sequences when adding. If False, expect
            single timesteps.
        add_batch_size: Batch size of experience added to buffer. If None,
            expects unbatched experience.
        priority_exponent: Priority exponent (alpha in PER paper). Controls how
            much prioritization is used. 0 = uniform sampling, 1 = full prioritization.
        device: Device for optimized operations ("cpu", "gpu", or "tpu").

    Returns:
        PrioritisedTrajectoryBuffer with episode-aware priority sampling.

    Example:
        >>> buffer = make_prioritised_episode_buffer(
        ...     max_length=100_000,
        ...     min_length=1000,
        ...     sample_batch_size=32,
        ...     sample_sequence_length=16,
        ...     add_batch_size=8,
        ...     priority_exponent=0.6,
        ... )
        >>> state = buffer.init(sample_transition)
        >>> state = buffer.add(state, transitions)
        >>> sample = buffer.sample(state, rng_key)
        >>> weights = compute_importance_weights(sample.probabilities, buffer_size, beta=0.4)
        >>> # After computing TD-errors:
        >>> state = buffer.set_priorities(state, sample.indices, jnp.abs(td_errors) + 1e-6)
    """
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
    validate_priority_exponent(priority_exponent)
    if not validate_device(device):
        device = "cpu"

    max_length_time_axis = max_length // add_batch_size
    min_length_time_axis = max(min_length // add_batch_size, sample_sequence_length)

    period = 1

    init_fn = functools.partial(
        prioritised_init,
        add_batch_size=add_batch_size,
        max_length_time_axis=max_length_time_axis,
        period=period,
    )

    add_fn = functools.partial(
        prioritised_episode_add,
        sample_sequence_length=sample_sequence_length,
        device=device,
    )

    if not add_batches:
        add_fn = add_dim_to_args(
            add_fn, axis=0, starting_arg_index=1, ending_arg_index=2
        )
    if not add_sequences:
        axis = 1 - int(not add_batches)
        add_fn = add_dim_to_args(
            add_fn, axis=axis, starting_arg_index=1, ending_arg_index=2
        )

    sample_fn = functools.partial(
        prioritised_episode_sample,
        sample_batch_size=sample_batch_size,
        sample_sequence_length=sample_sequence_length,
        get_start_flags=get_start_flags,
    )

    can_sample_fn = functools.partial(
        can_sample, min_length_time_axis=min_length_time_axis
    )

    set_priorities_fn = functools.partial(
        set_priorities, priority_exponent=priority_exponent, device=device
    )

    return PrioritisedTrajectoryBuffer(
        init=init_fn,
        add=add_fn,
        sample=sample_fn,
        can_sample=can_sample_fn,
        set_priorities=set_priorities_fn,
    )
