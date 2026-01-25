import distrax
import flax.linen as nn
import jax
import jax.numpy as jnp
import pytest

from memorax.networks.heads import (
    Alpha,
    Beta,
    C51QNetwork,
    Categorical,
    ContinuousQNetwork,
    DiscreteQNetwork,
    Gaussian,
    HLGaussVNetwork,
    SquashedGaussian,
    VNetwork,
)


class TestDiscreteQNetwork:
    """Test DiscreteQNetwork head."""

    @pytest.fixture
    def head(self):
        return DiscreteQNetwork(action_dim=4)

    @pytest.fixture
    def params(self, head, random_key):
        x = jnp.ones((2, 8))
        return head.init(random_key, x)

    def test_output_shape(self, head, params):
        x = jnp.ones((2, 8))
        q_values, aux = head.apply(params, x)
        assert q_values.shape == (2, 4)
        assert aux == {}

    def test_output_shape_batched(self, head, params):
        x = jnp.ones((4, 16, 8))
        q_values, aux = head.apply(params, x)
        assert q_values.shape == (4, 16, 4)

    def test_loss(self, head, params):
        x = jnp.ones((2, 8))
        q_values, aux = head.apply(params, x)
        targets = jnp.zeros((2, 4))
        loss = head.loss(q_values, aux, targets)
        assert loss.shape == ()
        assert loss >= 0

    def test_loss_zero_when_equal(self, head, params):
        x = jnp.ones((2, 8))
        q_values, aux = head.apply(params, x)
        loss = head.loss(q_values, aux, q_values)
        assert jnp.allclose(loss, 0.0)


class TestContinuousQNetwork:
    """Test ContinuousQNetwork head."""

    @pytest.fixture
    def head(self):
        return ContinuousQNetwork()

    @pytest.fixture
    def params(self, head, random_key):
        x = jnp.ones((2, 8))
        action = jnp.ones((2, 3))
        return head.init(random_key, x, action=action)

    def test_output_shape(self, head, params):
        x = jnp.ones((2, 8))
        action = jnp.ones((2, 3))
        q_values, aux = head.apply(params, x, action=action)
        assert q_values.shape == (2,)
        assert aux == {}

    def test_output_shape_batched(self, head, params):
        x = jnp.ones((4, 16, 8))
        action = jnp.ones((4, 16, 3))
        q_values, aux = head.apply(params, x, action=action)
        assert q_values.shape == (4, 16)

    def test_loss(self, head, params):
        x = jnp.ones((2, 8))
        action = jnp.ones((2, 3))
        q_values, aux = head.apply(params, x, action=action)
        targets = jnp.zeros((2,))
        loss = head.loss(q_values, aux, targets)
        assert loss.shape == ()
        assert loss >= 0

    def test_loss_zero_when_equal(self, head, params):
        x = jnp.ones((2, 8))
        action = jnp.ones((2, 3))
        q_values, aux = head.apply(params, x, action=action)
        loss = head.loss(q_values, aux, q_values)
        assert jnp.allclose(loss, 0.0)


class TestVNetwork:
    """Test VNetwork head."""

    @pytest.fixture
    def head(self):
        return VNetwork()

    @pytest.fixture
    def params(self, head, random_key):
        x = jnp.ones((2, 8))
        return head.init(random_key, x)

    def test_output_shape(self, head, params):
        x = jnp.ones((2, 8))
        v_value, aux = head.apply(params, x)
        assert v_value.shape == (2, 1)
        assert aux == {}

    def test_output_shape_batched(self, head, params):
        x = jnp.ones((4, 16, 8))
        v_value, aux = head.apply(params, x)
        assert v_value.shape == (4, 16, 1)

    def test_loss(self, head, params):
        x = jnp.ones((2, 8))
        v_value, aux = head.apply(params, x)
        targets = jnp.zeros((2, 1))
        loss = head.loss(v_value, aux, targets)
        assert loss.shape == ()
        assert loss >= 0

    def test_loss_zero_when_equal(self, head, params):
        x = jnp.ones((2, 8))
        v_value, aux = head.apply(params, x)
        loss = head.loss(v_value, aux, v_value)
        assert jnp.allclose(loss, 0.0)


class TestHLGaussVNetwork:
    """Test HLGaussVNetwork head."""

    @pytest.fixture
    def head(self):
        return HLGaussVNetwork(num_bins=51, v_min=-5.0, v_max=5.0)

    @pytest.fixture
    def params(self, head, random_key):
        x = jnp.ones((2, 8))
        return head.init(random_key, x)

    def test_output_shape(self, head, params):
        x = jnp.ones((2, 8))
        v_value, aux = head.apply(params, x)
        assert v_value.shape == (2, 1)
        assert "logits" in aux
        assert aux["logits"].shape == (2, 51)

    def test_output_shape_batched(self, head, params):
        x = jnp.ones((4, 16, 8))
        v_value, aux = head.apply(params, x)
        assert v_value.shape == (4, 16, 1)
        assert aux["logits"].shape == (4, 16, 51)

    def test_value_in_range(self, head, params):
        x = jnp.ones((2, 8))
        v_value, aux = head.apply(params, x)
        assert jnp.all(v_value >= -5.0)
        assert jnp.all(v_value <= 5.0)

    def test_loss(self, head, params):
        x = jnp.ones((2, 8))
        v_value, aux = head.apply(params, x)
        targets = jnp.zeros((2, 1))
        loss = head.loss(v_value, aux, targets)
        assert loss.shape == ()
        assert loss >= 0

    def test_loss_clamps_targets(self, head, params):
        x = jnp.ones((2, 8))
        v_value, aux = head.apply(params, x)
        # Targets outside range should be clamped
        targets = jnp.array([[100.0], [-100.0]])
        loss = head.loss(v_value, aux, targets)
        assert loss.shape == ()
        assert jnp.isfinite(loss)

    def test_bin_centers(self, head, params):
        # Verify bin centers are correctly computed (access via bind to enter module scope)
        bound_head = head.bind(params)
        assert bound_head.bin_centers.shape == (51,)
        assert jnp.allclose(bound_head.bin_centers[0], -5.0)
        assert jnp.allclose(bound_head.bin_centers[-1], 5.0)


class TestC51QNetwork:

    @pytest.fixture
    def head(self):
        return C51QNetwork(action_dim=4, num_atoms=51, v_min=-5.0, v_max=5.0)

    @pytest.fixture
    def params(self, head, random_key):
        x = jnp.ones((2, 8))
        return head.init(random_key, x)

    def test_output_shape(self, head, params):
        x = jnp.ones((2, 8))
        q_values, aux = head.apply(params, x)
        assert q_values.shape == (2, 4)
        assert "logits" in aux
        assert "probs" in aux
        assert aux["logits"].shape == (2, 4, 51)
        assert aux["probs"].shape == (2, 4, 51)

    def test_output_shape_batched(self, head, params):
        x = jnp.ones((4, 16, 8))
        q_values, aux = head.apply(params, x)
        assert q_values.shape == (4, 16, 4)
        assert aux["logits"].shape == (4, 16, 4, 51)
        assert aux["probs"].shape == (4, 16, 4, 51)

    def test_q_values_in_range(self, head, params):
        x = jnp.ones((2, 8))
        q_values, aux = head.apply(params, x)
        assert jnp.all(q_values >= -5.0)
        assert jnp.all(q_values <= 5.0)

    def test_probs_sum_to_one(self, head, params):
        x = jnp.ones((2, 8))
        _, aux = head.apply(params, x)
        prob_sums = aux["probs"].sum(axis=-1)
        assert jnp.allclose(prob_sums, 1.0)

    def test_loss(self, head, params):
        x = jnp.ones((2, 8))
        q_values, aux = head.apply(params, x)
        targets = jnp.zeros((2, 4))
        loss = head.loss(q_values, aux, targets)
        assert loss.shape == ()
        assert loss >= 0

    def test_loss_clamps_targets(self, head, params):
        x = jnp.ones((2, 8))
        q_values, aux = head.apply(params, x)
        targets = jnp.array([[100.0, -100.0, 0.0, 0.0], [0.0, 0.0, 100.0, -100.0]])
        loss = head.loss(q_values, aux, targets)
        assert loss.shape == ()
        assert jnp.isfinite(loss)

    def test_atoms(self, head, params):
        bound_head = head.bind(params)
        assert bound_head.atoms.shape == (51,)
        assert jnp.allclose(bound_head.atoms[0], -5.0)
        assert jnp.allclose(bound_head.atoms[-1], 5.0)


class TestCategorical:

    @pytest.fixture
    def head(self):
        return Categorical(action_dim=4)

    @pytest.fixture
    def params(self, head, random_key):
        x = jnp.ones((2, 8))
        return head.init(random_key, x)

    def test_output_type(self, head, params):
        x = jnp.ones((2, 8))
        dist, aux = head.apply(params, x)
        assert isinstance(dist, distrax.Categorical)
        assert aux == {}

    def test_sample_shape(self, head, params, random_key):
        x = jnp.ones((2, 8))
        dist, aux = head.apply(params, x)
        samples = dist.sample(seed=random_key)
        assert samples.shape == (2,)
        assert samples.dtype == jnp.int32

    def test_log_prob_shape(self, head, params, random_key):
        x = jnp.ones((2, 8))
        dist, aux = head.apply(params, x)
        actions = jnp.array([0, 1])
        log_probs = dist.log_prob(actions)
        assert log_probs.shape == (2,)

    def test_entropy_shape(self, head, params):
        x = jnp.ones((2, 8))
        dist, aux = head.apply(params, x)
        entropy = dist.entropy()
        assert entropy.shape == (2,)
        assert jnp.all(entropy >= 0)

    def test_batched_shape(self, head, params, random_key):
        x = jnp.ones((4, 16, 8))
        dist, aux = head.apply(params, x)
        samples = dist.sample(seed=random_key)
        assert samples.shape == (4, 16)


class TestGaussian:
    """Test Gaussian distribution head."""

    @pytest.fixture
    def head(self):
        return Gaussian(action_dim=3)

    @pytest.fixture
    def params(self, head, random_key):
        x = jnp.ones((2, 8))
        return head.init(random_key, x)

    def test_output_type(self, head, params):
        x = jnp.ones((2, 8))
        dist, aux = head.apply(params, x)
        assert isinstance(dist, distrax.Transformed)
        assert aux == {}

    def test_sample_shape(self, head, params, random_key):
        x = jnp.ones((2, 8))
        dist, aux = head.apply(params, x)
        samples = dist.sample(seed=random_key)
        assert samples.shape == (2, 3)

    def test_log_prob_shape(self, head, params):
        x = jnp.ones((2, 8))
        dist, aux = head.apply(params, x)
        actions = jnp.zeros((2, 3))
        log_probs = dist.log_prob(actions)
        assert log_probs.shape == (2,)

    def test_entropy_shape(self, head, params):
        x = jnp.ones((2, 8))
        dist, aux = head.apply(params, x)
        entropy = dist.entropy()
        assert entropy.shape == (2,)

    def test_batched_shape(self, head, params, random_key):
        x = jnp.ones((4, 16, 8))
        dist, aux = head.apply(params, x)
        samples = dist.sample(seed=random_key)
        assert samples.shape == (4, 16, 3)

    def test_custom_transform(self, random_key):
        # Test with a custom transform (tanh)
        head = Gaussian(action_dim=3, transform=jnp.tanh)
        x = jnp.ones((2, 8))
        params = head.init(random_key, x)
        dist, aux = head.apply(params, x)
        samples = dist.sample(seed=random_key)
        assert samples.shape == (2, 3)
        # Tanh outputs should be in [-1, 1]
        assert jnp.all(samples >= -1.0)
        assert jnp.all(samples <= 1.0)


class TestSquashedGaussian:
    """Test SquashedGaussian distribution head."""

    @pytest.fixture
    def head(self):
        return SquashedGaussian(action_dim=3)

    @pytest.fixture
    def params(self, head, random_key):
        x = jnp.ones((2, 8))
        return head.init(random_key, x)

    def test_output_type(self, head, params):
        x = jnp.ones((2, 8))
        dist, aux = head.apply(params, x)
        assert isinstance(dist, distrax.Transformed)
        assert aux == {}

    def test_sample_shape(self, head, params, random_key):
        x = jnp.ones((2, 8))
        dist, aux = head.apply(params, x)
        samples = dist.sample(seed=random_key)
        assert samples.shape == (2, 3)

    def test_samples_bounded(self, head, params, random_key):
        x = jnp.ones((2, 8))
        dist, aux = head.apply(params, x)
        # Sample multiple times to check bounds
        key = random_key
        for _ in range(10):
            key, subkey = jax.random.split(key)
            samples = dist.sample(seed=subkey)
            assert jnp.all(samples >= -1.0)
            assert jnp.all(samples <= 1.0)

    def test_log_prob_shape(self, head, params):
        x = jnp.ones((2, 8))
        dist, aux = head.apply(params, x)
        actions = jnp.zeros((2, 3))
        log_probs = dist.log_prob(actions)
        assert log_probs.shape == (2,)

    def test_batched_shape(self, head, params, random_key):
        x = jnp.ones((4, 16, 8))
        dist, aux = head.apply(params, x)
        samples = dist.sample(seed=random_key)
        assert samples.shape == (4, 16, 3)

    def test_temperature(self, head, params, random_key):
        x = jnp.ones((2, 8))
        # Default temperature
        dist1, _ = head.apply(params, x)
        # Lower temperature (should have lower variance)
        dist2, _ = head.apply(params, x, temperature=0.1)

        # Sample many times and compare variances
        key = random_key
        samples1 = []
        samples2 = []
        for _ in range(100):
            key, k1, k2 = jax.random.split(key, 3)
            samples1.append(dist1.sample(seed=k1))
            samples2.append(dist2.sample(seed=k2))

        var1 = jnp.var(jnp.stack(samples1), axis=0).mean()
        var2 = jnp.var(jnp.stack(samples2), axis=0).mean()
        # Lower temperature should have lower variance
        assert var2 < var1

    def test_log_std_clipping(self, head, params):
        # Verify log_std is clipped to valid range
        x = jnp.ones((2, 8))
        dist, aux = head.apply(params, x)
        # The distribution should produce finite log probs
        actions = jnp.zeros((2, 3))
        log_probs = dist.log_prob(actions)
        assert jnp.all(jnp.isfinite(log_probs))


class TestAlpha:
    """Test Alpha learnable parameter."""

    @pytest.fixture
    def head(self):
        return Alpha(initial_alpha=1.0)

    @pytest.fixture
    def params(self, head, random_key):
        return head.init(random_key)

    def test_output_shape(self, head, params):
        log_alpha = head.apply(params)
        assert log_alpha.shape == ()

    def test_initial_value(self, head, params):
        log_alpha = head.apply(params)
        alpha = jnp.exp(log_alpha)
        assert jnp.allclose(alpha, 1.0)

    def test_different_initial_values(self, random_key):
        for initial_alpha in [0.1, 0.5, 2.0, 10.0]:
            head = Alpha(initial_alpha=initial_alpha)
            params = head.init(random_key)
            log_alpha = head.apply(params)
            alpha = jnp.exp(log_alpha)
            assert jnp.allclose(alpha, initial_alpha, rtol=1e-5)


class TestBeta:
    """Test Beta learnable parameter."""

    @pytest.fixture
    def head(self):
        return Beta(initial_beta=1.0)

    @pytest.fixture
    def params(self, head, random_key):
        return head.init(random_key)

    def test_output_shape(self, head, params):
        log_beta = head.apply(params)
        assert log_beta.shape == ()

    def test_initial_value(self, head, params):
        log_beta = head.apply(params)
        beta = jnp.exp(log_beta)
        assert jnp.allclose(beta, 1.0)

    def test_different_initial_values(self, random_key):
        for initial_beta in [0.1, 0.5, 2.0, 10.0]:
            head = Beta(initial_beta=initial_beta)
            params = head.init(random_key)
            log_beta = head.apply(params)
            beta = jnp.exp(log_beta)
            assert jnp.allclose(beta, initial_beta, rtol=1e-5)
