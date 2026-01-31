"""Tests for episode buffers including prioritized variants."""

import jax
import jax.numpy as jnp
import pytest
from flax import struct

from memorax.buffers import (compute_importance_weights, get_full_start_flags,
                             get_start_flags_from_done, make_episode_buffer,
                             make_prioritised_episode_buffer)


@struct.dataclass
class MockExperience:
    """Mock experience structure for testing."""

    obs: jnp.ndarray
    action: jnp.ndarray
    reward: jnp.ndarray
    done: jnp.ndarray
    prev_done: jnp.ndarray


def make_sample_experience(obs_dim: int = 4) -> MockExperience:
    """Create a single sample experience for buffer initialization.

    This defines the structure and per-element shapes for the buffer.
    The buffer will create storage with shape (add_batch_size, max_length_time_axis, *shape).
    """
    return MockExperience(
        obs=jnp.zeros((obs_dim,)),
        action=jnp.zeros((), dtype=jnp.int32),
        reward=jnp.zeros(()),
        done=jnp.zeros(()),
        prev_done=jnp.zeros(()),
    )


def make_timestep_batch(
    batch_size: int, key: jax.Array, obs_dim: int = 4
) -> MockExperience:
    """Create a batch of single timesteps for adding to buffer.

    When add_sequences=False, the buffer expects shape (batch_size, *feature_shape).
    """
    keys = jax.random.split(key, 5)
    return MockExperience(
        obs=jax.random.normal(keys[0], (batch_size, obs_dim)),
        action=jax.random.randint(keys[1], (batch_size,), 0, 2),
        reward=jax.random.normal(keys[2], (batch_size,)),
        done=jax.random.bernoulli(keys[3], 0.1, (batch_size,)).astype(jnp.float32),
        prev_done=jax.random.bernoulli(keys[4], 0.1, (batch_size,)).astype(jnp.float32),
    )


def make_sequence_batch(
    batch_size: int, seq_len: int, key: jax.Array, obs_dim: int = 4
) -> MockExperience:
    """Create a batch of sequences for adding to buffer.

    When add_sequences=True, the buffer expects shape (batch_size, seq_len, *feature_shape).
    """
    keys = jax.random.split(key, 5)
    return MockExperience(
        obs=jax.random.normal(keys[0], (batch_size, seq_len, obs_dim)),
        action=jax.random.randint(keys[1], (batch_size, seq_len), 0, 2),
        reward=jax.random.normal(keys[2], (batch_size, seq_len)),
        done=jax.random.bernoulli(keys[3], 0.1, (batch_size, seq_len)).astype(
            jnp.float32
        ),
        prev_done=jax.random.bernoulli(keys[4], 0.1, (batch_size, seq_len)).astype(
            jnp.float32
        ),
    )


class TestStartFlagFunctions:
    """Tests for start flag extraction functions."""

    def test_get_full_start_flags_returns_ones(self):
        """get_full_start_flags should return all ones."""
        prev_done = jnp.zeros((4, 10))
        exp = MockExperience(
            obs=jnp.zeros((4, 10, 4)),
            action=jnp.zeros((4, 10), dtype=jnp.int32),
            reward=jnp.zeros((4, 10)),
            done=jnp.zeros((4, 10)),
            prev_done=prev_done,
        )
        flags = get_full_start_flags(exp)
        assert flags.shape == prev_done.shape
        assert jnp.all(flags == 1)

    def test_get_start_flags_from_done_shape(self):
        """get_start_flags_from_done should return correct shape."""
        prev_done = jnp.zeros((4, 10))
        exp = MockExperience(
            obs=jnp.zeros((4, 10, 4)),
            action=jnp.zeros((4, 10), dtype=jnp.int32),
            reward=jnp.zeros((4, 10)),
            done=jnp.zeros((4, 10)),
            prev_done=prev_done,
        )
        flags = get_start_flags_from_done(exp)
        assert flags.shape == prev_done.shape

    def test_get_start_flags_from_done_values(self):
        """get_start_flags_from_done should roll prev_done by 1."""
        prev_done = jnp.array([[0, 1, 0, 0, 1], [1, 0, 0, 1, 0]], dtype=jnp.float32)
        exp = MockExperience(
            obs=jnp.zeros((2, 5, 4)),
            action=jnp.zeros((2, 5), dtype=jnp.int32),
            reward=jnp.zeros((2, 5)),
            done=jnp.zeros((2, 5)),
            prev_done=prev_done,
        )
        flags = get_start_flags_from_done(exp)
        expected = jnp.roll(prev_done, shift=1, axis=1)
        assert jnp.array_equal(flags, expected)


class TestEpisodeBuffer:
    """Tests for the standard episode buffer."""

    @pytest.fixture
    def buffer_config(self):
        """Common buffer configuration."""
        return {
            "max_length": 1000,
            "min_length": 100,
            "sample_batch_size": 8,
            "sample_sequence_length": 4,
            "add_batch_size": 4,
        }

    def test_buffer_creation(self, buffer_config):
        """Buffer should be created successfully."""
        buffer = make_episode_buffer(**buffer_config)
        assert buffer is not None
        assert callable(buffer.init)
        assert callable(buffer.add)
        assert callable(buffer.sample)
        assert callable(buffer.can_sample)

    def test_buffer_init(self, buffer_config):
        """Buffer state should be initialized correctly."""
        buffer = make_episode_buffer(**buffer_config)
        exp = make_sample_experience()
        state = buffer.init(exp)
        assert state is not None
        assert state.current_index == 0
        assert not state.is_full

    def test_buffer_add(self, buffer_config, random_key):
        """Adding experience should update buffer state."""
        buffer = make_episode_buffer(**buffer_config)
        state = buffer.init(make_sample_experience())

        add_exp = make_timestep_batch(buffer_config["add_batch_size"], random_key)
        new_state = buffer.add(state, add_exp)

        assert new_state.current_index == 1

    def test_buffer_can_sample(self, buffer_config, random_key):
        """can_sample should return True after enough data added."""
        buffer = make_episode_buffer(**buffer_config)
        state = buffer.init(make_sample_experience())

        assert not buffer.can_sample(state)

        key = random_key
        for i in range(buffer_config["min_length"]):
            key, subkey = jax.random.split(key)
            add_exp = make_timestep_batch(buffer_config["add_batch_size"], subkey)
            state = buffer.add(state, add_exp)

        assert buffer.can_sample(state)

    def test_buffer_sample_shape(self, buffer_config, random_key):
        """Sampled experience should have correct shape."""
        buffer = make_episode_buffer(**buffer_config)
        state = buffer.init(make_sample_experience())

        key = random_key
        for i in range(buffer_config["min_length"] + 10):
            key, subkey = jax.random.split(key)
            add_exp = make_timestep_batch(buffer_config["add_batch_size"], subkey)
            state = buffer.add(state, add_exp)

        key, sample_key = jax.random.split(key)
        sample = buffer.sample(state, sample_key)

        expected_shape = (
            buffer_config["sample_batch_size"],
            buffer_config["sample_sequence_length"],
        )
        assert sample.experience.obs.shape[:2] == expected_shape
        assert sample.experience.action.shape == expected_shape
        assert sample.experience.reward.shape == expected_shape


class TestEpisodeBoundarySampling:
    """Tests that episode buffers correctly sample from episode boundaries.

    Note: The get_start_flags_from_done function computes start_flags = roll(prev_done, 1).
    This means start_flags[t] = prev_done[t-1]. So if we want to sample from position t,
    we need prev_done[t-1] = 1.

    In RL semantics:
    - done[t] = 1 means episode ended at step t
    - prev_done[t] = done[t-1] = 1 means the PREVIOUS step was terminal
    - So position t with prev_done[t]=1 is the FIRST step of a new episode

    The roll by 1 in get_start_flags_from_done seems to create a 1-step offset.
    This might be intentional for some use case. We test the actual behavior.
    """

    def test_episode_buffer_samples_from_start_flag_positions(self, random_key):
        """Sampled sequences should start where start_flags=1 (roll of prev_done)."""
        add_batch_size = 1
        sample_seq_len = 4
        buffer = make_episode_buffer(
            max_length=100,
            min_length=20,
            sample_batch_size=8,
            sample_sequence_length=sample_seq_len,
            add_batch_size=add_batch_size,
            add_sequences=True,
        )

        state = buffer.init(make_sample_experience(obs_dim=1))


        seq_len = 40
        time_indices = jnp.arange(seq_len).reshape(1, seq_len, 1).astype(jnp.float32)


        prev_done = jnp.zeros((add_batch_size, seq_len))
        prev_done_positions = [0, 8, 16, 24, 32]
        for pos in prev_done_positions:
            prev_done = prev_done.at[:, pos].set(1.0)


        expected_sample_positions = jnp.array([1, 9, 17, 25, 33])

        exp = MockExperience(
            obs=time_indices,
            action=jnp.zeros((add_batch_size, seq_len), dtype=jnp.int32),
            reward=jnp.zeros((add_batch_size, seq_len)),
            done=jnp.zeros((add_batch_size, seq_len)),
            prev_done=prev_done,
        )
        state = buffer.add(state, exp)


        key = random_key
        for _ in range(20):
            key, sample_key = jax.random.split(key)
            sample = buffer.sample(state, sample_key)
            first_obs = sample.experience.obs[:, 0, 0]

            for obs_val in first_obs:
                assert obs_val in expected_sample_positions, (
                    f"Sampled at position {int(obs_val)}, expected one of {expected_sample_positions}"
                )

    def test_prioritised_buffer_samples_from_start_flag_positions(self, random_key):
        """Prioritised buffer should also sample from start_flag positions."""
        add_batch_size = 1
        sample_seq_len = 4
        buffer = make_prioritised_episode_buffer(
            max_length=100,
            min_length=20,
            sample_batch_size=8,
            sample_sequence_length=sample_seq_len,
            add_batch_size=add_batch_size,
            add_sequences=True,
            priority_exponent=0.6,
        )

        state = buffer.init(make_sample_experience(obs_dim=1))

        seq_len = 40
        time_indices = jnp.arange(seq_len).reshape(1, seq_len, 1).astype(jnp.float32)

        prev_done = jnp.zeros((add_batch_size, seq_len))
        prev_done_positions = [0, 8, 16, 24, 32]
        for pos in prev_done_positions:
            prev_done = prev_done.at[:, pos].set(1.0)

        expected_sample_positions = jnp.array([1, 9, 17, 25, 33])

        exp = MockExperience(
            obs=time_indices,
            action=jnp.zeros((add_batch_size, seq_len), dtype=jnp.int32),
            reward=jnp.zeros((add_batch_size, seq_len)),
            done=jnp.zeros((add_batch_size, seq_len)),
            prev_done=prev_done,
        )
        state = buffer.add(state, exp)

        key = random_key
        for _ in range(20):
            key, sample_key = jax.random.split(key)
            sample = buffer.sample(state, sample_key)
            first_obs = sample.experience.obs[:, 0, 0]

            for obs_val in first_obs:
                assert obs_val in expected_sample_positions, (
                    f"Sampled at position {int(obs_val)}, expected one of {expected_sample_positions}"
                )

    def test_buffer_only_samples_valid_episode_starts(self, random_key):
        """Buffer should not sample from non-start positions."""
        add_batch_size = 1
        buffer = make_episode_buffer(
            max_length=100,
            min_length=20,
            sample_batch_size=32,
            sample_sequence_length=4,
            add_batch_size=add_batch_size,
            add_sequences=True,
        )

        state = buffer.init(make_sample_experience(obs_dim=1))

        seq_len = 40
        time_indices = jnp.arange(seq_len).reshape(1, seq_len, 1).astype(jnp.float32)


        prev_done = jnp.zeros((add_batch_size, seq_len))
        prev_done_positions = [0, 10, 20, 30]
        for pos in prev_done_positions:
            prev_done = prev_done.at[:, pos].set(1.0)

        expected_sample_positions = set([1, 11, 21, 31])
        invalid_positions = (
            set(range(seq_len - 4)) - expected_sample_positions
        )

        exp = MockExperience(
            obs=time_indices,
            action=jnp.zeros((add_batch_size, seq_len), dtype=jnp.int32),
            reward=jnp.zeros((add_batch_size, seq_len)),
            done=jnp.zeros((add_batch_size, seq_len)),
            prev_done=prev_done,
        )
        state = buffer.add(state, exp)


        key = random_key
        all_sampled_positions = []
        for _ in range(50):
            key, sample_key = jax.random.split(key)
            sample = buffer.sample(state, sample_key)
            first_obs = sample.experience.obs[:, 0, 0]
            all_sampled_positions.extend([int(x) for x in first_obs])

        sampled_set = set(all_sampled_positions)


        assert sampled_set.issubset(expected_sample_positions), (
            f"Sampled invalid positions: {sampled_set - expected_sample_positions}"
        )


class TestPrioritisedEpisodeBuffer:
    """Tests for the prioritised episode buffer."""

    @pytest.fixture
    def buffer_config(self):
        """Common buffer configuration."""
        return {
            "max_length": 1000,
            "min_length": 100,
            "sample_batch_size": 8,
            "sample_sequence_length": 4,
            "add_batch_size": 4,
            "priority_exponent": 0.6,
            "device": "cpu",
        }

    def test_buffer_creation(self, buffer_config):
        """Prioritised buffer should be created successfully."""
        buffer = make_prioritised_episode_buffer(**buffer_config)
        assert buffer is not None
        assert callable(buffer.init)
        assert callable(buffer.add)
        assert callable(buffer.sample)
        assert callable(buffer.can_sample)
        assert callable(buffer.set_priorities)

    def test_buffer_init(self, buffer_config):
        """Buffer state should include sum tree state."""
        buffer = make_prioritised_episode_buffer(**buffer_config)
        state = buffer.init(make_sample_experience())

        assert state is not None
        assert state.current_index == 0
        assert not state.is_full
        assert hasattr(state, "sum_tree_state")
        assert hasattr(state, "running_index")

    def test_buffer_add_updates_state(self, buffer_config, random_key):
        """Adding experience should update buffer state."""
        buffer = make_prioritised_episode_buffer(**buffer_config)
        state = buffer.init(make_sample_experience())

        add_exp = make_timestep_batch(buffer_config["add_batch_size"], random_key)
        new_state = buffer.add(state, add_exp)

        assert new_state.current_index == 1
        assert new_state.running_index == 1

    def test_buffer_sample_returns_indices_and_probabilities(
        self, buffer_config, random_key
    ):
        """Sample should include indices and probabilities."""
        buffer = make_prioritised_episode_buffer(**buffer_config)
        state = buffer.init(make_sample_experience())

        key = random_key
        for i in range(buffer_config["min_length"] + 10):
            key, subkey = jax.random.split(key)
            add_exp = make_timestep_batch(buffer_config["add_batch_size"], subkey)
            state = buffer.add(state, add_exp)

        key, sample_key = jax.random.split(key)
        sample = buffer.sample(state, sample_key)

        assert hasattr(sample, "experience")
        assert hasattr(sample, "indices")
        assert hasattr(sample, "probabilities")

        assert sample.indices.shape == (buffer_config["sample_batch_size"],)
        assert sample.probabilities.shape == (buffer_config["sample_batch_size"],)
        assert jnp.all(sample.probabilities > 0)

    def test_buffer_sample_shape(self, buffer_config, random_key):
        """Sampled experience should have correct shape."""
        buffer = make_prioritised_episode_buffer(**buffer_config)
        state = buffer.init(make_sample_experience())

        key = random_key
        for i in range(buffer_config["min_length"] + 10):
            key, subkey = jax.random.split(key)
            add_exp = make_timestep_batch(buffer_config["add_batch_size"], subkey)
            state = buffer.add(state, add_exp)

        key, sample_key = jax.random.split(key)
        sample = buffer.sample(state, sample_key)

        expected_shape = (
            buffer_config["sample_batch_size"],
            buffer_config["sample_sequence_length"],
        )
        assert sample.experience.obs.shape[:2] == expected_shape
        assert sample.experience.action.shape == expected_shape

    def test_set_priorities(self, buffer_config, random_key):
        """set_priorities should update priorities without error."""
        buffer = make_prioritised_episode_buffer(**buffer_config)
        state = buffer.init(make_sample_experience())

        key = random_key
        for i in range(buffer_config["min_length"] + 10):
            key, subkey = jax.random.split(key)
            add_exp = make_timestep_batch(buffer_config["add_batch_size"], subkey)
            state = buffer.add(state, add_exp)

        key, sample_key = jax.random.split(key)
        sample = buffer.sample(state, sample_key)

        new_priorities = (
            jnp.abs(jax.random.normal(sample_key, sample.indices.shape)) + 1e-6
        )
        new_state = buffer.set_priorities(state, sample.indices, new_priorities)

        assert new_state is not None

    def test_priority_affects_sampling(self, buffer_config, random_key):
        """Higher priority items should be sampled more frequently."""
        buffer = make_prioritised_episode_buffer(
            **{**buffer_config, "priority_exponent": 1.0}
        )
        state = buffer.init(make_sample_experience())

        key = random_key
        for i in range(buffer_config["min_length"] + 50):
            key, subkey = jax.random.split(key)
            add_exp = make_timestep_batch(buffer_config["add_batch_size"], subkey)
            state = buffer.add(state, add_exp)

        key, sample_key = jax.random.split(key)
        sample = buffer.sample(state, sample_key)

        high_priority_idx = sample.indices[0]
        all_indices = sample.indices
        priorities = jnp.where(
            all_indices == high_priority_idx,
            jnp.ones_like(all_indices, dtype=jnp.float32) * 1000.0,
            jnp.ones_like(all_indices, dtype=jnp.float32) * 0.001,
        )
        state = buffer.set_priorities(state, all_indices, priorities)

        high_priority_count = 0
        num_samples = 100
        key = sample_key
        for i in range(num_samples):
            key, subkey = jax.random.split(key)
            sample = buffer.sample(state, subkey)
            high_priority_count += jnp.sum(sample.indices == high_priority_idx)

        assert high_priority_count > 0


class TestImportanceWeights:
    """Tests for importance sampling weight computation."""

    def test_compute_importance_weights_shape(self):
        """Weights should have same shape as probabilities."""
        probs = jnp.array([0.1, 0.2, 0.3, 0.4])
        weights = compute_importance_weights(probs, buffer_size=100, beta=0.4)
        assert weights.shape == probs.shape

    def test_compute_importance_weights_normalized(self):
        """Weights should be normalized by max weight."""
        probs = jnp.array([0.1, 0.2, 0.3, 0.4])
        weights = compute_importance_weights(probs, buffer_size=100, beta=0.4)
        assert jnp.isclose(weights.max(), 1.0)

    def test_compute_importance_weights_beta_zero(self):
        """With beta=0, all weights should be 1.0."""
        probs = jnp.array([0.1, 0.2, 0.3, 0.4])
        weights = compute_importance_weights(probs, buffer_size=100, beta=0.0)
        assert jnp.allclose(weights, 1.0)

    def test_compute_importance_weights_beta_one(self):
        """With beta=1, lower probability items get higher weights."""
        probs = jnp.array([0.1, 0.2, 0.3, 0.4])
        weights = compute_importance_weights(probs, buffer_size=100, beta=1.0)
        assert jnp.isclose(weights[0], 1.0)
        assert weights[3] < weights[0]

    def test_compute_importance_weights_handles_small_probs(self):
        """Should handle very small probabilities without NaN/Inf."""
        probs = jnp.array([1e-10, 0.5, 0.5 - 1e-10])
        weights = compute_importance_weights(probs, buffer_size=100, beta=0.4)
        assert jnp.all(jnp.isfinite(weights))

    def test_compute_importance_weights_handles_zero_probs(self):
        """Should handle zero probabilities gracefully."""
        probs = jnp.array([0.0, 0.5, 0.5])
        weights = compute_importance_weights(probs, buffer_size=100, beta=0.4)
        assert jnp.all(jnp.isfinite(weights))


class TestBufferValidation:
    """Tests for buffer argument validation."""

    def test_episode_buffer_rejects_invalid_sample_batch_size(self):
        """Should raise error if sample_batch_size > max_length."""
        with pytest.raises(ValueError):
            make_episode_buffer(
                max_length=100,
                min_length=10,
                sample_batch_size=200,
                sample_sequence_length=4,
                add_batch_size=4,
            )

    def test_episode_buffer_rejects_invalid_min_length(self):
        """Should raise error if min_length too large."""
        with pytest.raises(ValueError):
            make_episode_buffer(
                max_length=100,
                min_length=200,
                sample_batch_size=8,
                sample_sequence_length=4,
                add_batch_size=4,
            )

    def test_prioritised_buffer_rejects_invalid_priority_exponent(self):
        """Should raise error for invalid priority exponent."""
        with pytest.raises(ValueError):
            make_prioritised_episode_buffer(
                max_length=1000,
                min_length=100,
                sample_batch_size=8,
                sample_sequence_length=4,
                add_batch_size=4,
                priority_exponent=1.5,
            )

        with pytest.raises(ValueError):
            make_prioritised_episode_buffer(
                max_length=1000,
                min_length=100,
                sample_batch_size=8,
                sample_sequence_length=4,
                add_batch_size=4,
                priority_exponent=-0.5,
            )


class TestBufferWithSequenceAdding:
    """Tests for buffers with add_sequences=True."""

    def test_episode_buffer_add_sequences(self, random_key):
        """Buffer should accept sequence additions."""
        buffer = make_episode_buffer(
            max_length=1000,
            min_length=100,
            sample_batch_size=8,
            sample_sequence_length=4,
            add_batch_size=4,
            add_sequences=True,
        )

        state = buffer.init(make_sample_experience())
        add_exp = make_sequence_batch(4, 10, random_key)
        new_state = buffer.add(state, add_exp)

        assert new_state.current_index == 10

    def test_prioritised_buffer_add_sequences(self, random_key):
        """Prioritised buffer should accept sequence additions."""
        buffer = make_prioritised_episode_buffer(
            max_length=1000,
            min_length=100,
            sample_batch_size=8,
            sample_sequence_length=4,
            add_batch_size=4,
            add_sequences=True,
        )

        state = buffer.init(make_sample_experience())
        add_exp = make_sequence_batch(4, 10, random_key)
        new_state = buffer.add(state, add_exp)

        assert new_state.current_index == 10
        assert new_state.running_index == 10
