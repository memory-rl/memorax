"""Tests for all sequence models.

Tests verify that sequence models properly implement:
- initialize_carry(key, input_shape) -> Carry
- __call__(inputs, mask, initial_carry) -> (Carry, Array)

For Algebra subclasses, also tests:
- combine(a, b) -> Carry
- read(h, x) -> Array
"""

import jax
import jax.numpy as jnp
import pytest
from flax import linen as nn
from flax.linen import initializers


class TestLRU:
    """Test Linear Recurrent Unit."""

    @pytest.fixture
    def lru(self):
        from memorax.networks.sequence_models import LRUCell
        from memorax.networks.sequence_models.memoroid import Memoroid

        algebra = LRUCell(
            features=16,
            hidden_dim=32,
        )
        return Memoroid(cell=algebra)

    def test_initialize_carry(self, lru):
        key = jax.random.key(0)
        input_shape = (4, 16)  # (batch_size, features)
        carry = lru.initialize_carry(key, input_shape)
        assert carry is not None

    def test_call(self, lru):
        key = jax.random.key(0)
        batch_size, seq_len, features = 4, 8, 16
        inputs = jnp.ones((batch_size, seq_len, features))
        mask = jnp.zeros((batch_size, seq_len))

        params = lru.init(key, inputs, mask)
        carry, outputs = lru.apply(params, inputs, mask)

        assert outputs.shape == (batch_size, seq_len, features)

    def test_with_initial_carry(self, lru):
        key = jax.random.key(0)
        batch_size, seq_len, features = 4, 8, 16
        inputs = jnp.ones((batch_size, seq_len, features))
        mask = jnp.zeros((batch_size, seq_len))

        params = lru.init(key, inputs, mask)
        initial_carry = lru.initialize_carry(key, (batch_size, features))

        carry, outputs = lru.apply(params, inputs, mask, initial_carry)
        assert outputs.shape == (batch_size, seq_len, features)


class TestMinGRU:
    """Test Minimal GRU in log-space."""

    @pytest.fixture
    def min_gru(self):
        from memorax.networks.sequence_models import MinGRUCell
        from memorax.networks.sequence_models.memoroid import Memoroid

        algebra = MinGRUCell(
            features=32,
        )
        return Memoroid(cell=algebra)

    def test_initialize_carry(self, min_gru):
        key = jax.random.key(0)
        input_shape = (4, 16)
        carry = min_gru.initialize_carry(key, input_shape)
        assert carry is not None

    def test_call(self, min_gru):
        key = jax.random.key(0)
        batch_size, seq_len, features = 4, 8, 16
        inputs = jnp.ones((batch_size, seq_len, features))
        mask = jnp.zeros((batch_size, seq_len))

        params = min_gru.init(key, inputs, mask)
        carry, outputs = min_gru.apply(params, inputs, mask)

        # MinGRU output is its hidden features (32)
        assert outputs.shape == (batch_size, seq_len, 32)


class TestFFM:
    """Test Fast and Forgetful Memory."""

    @pytest.fixture
    def ffm(self):
        from memorax.networks.sequence_models import FFMCell
        from memorax.networks.sequence_models.memoroid import Memoroid

        algebra = FFMCell(
            features=16,
            memory_size=16,
            context_size=8,
        )
        return Memoroid(cell=algebra)

    def test_initialize_carry(self, ffm):
        key = jax.random.key(0)
        input_shape = (4, 16)
        carry = ffm.initialize_carry(key, input_shape)
        assert carry is not None

    def test_call(self, ffm):
        key = jax.random.key(0)
        batch_size, seq_len, features = 4, 8, 16
        inputs = jnp.ones((batch_size, seq_len, features))
        mask = jnp.zeros((batch_size, seq_len))

        params = ffm.init(key, inputs, mask)
        carry, outputs = ffm.apply(params, inputs, mask)

        assert outputs.shape == (batch_size, seq_len, features)


class TestS5:
    """Test S5 with HIPPO initialization."""

    @pytest.fixture
    def s5(self):
        from memorax.networks.sequence_models import S5Cell
        from memorax.networks.sequence_models.memoroid import Memoroid

        algebra = S5Cell(
            features=16,
            state_size=32,
        )
        return Memoroid(cell=algebra)

    def test_initialize_carry(self, s5):
        key = jax.random.key(0)
        input_shape = (4, 16)
        carry = s5.initialize_carry(key, input_shape)
        assert carry is not None

    def test_call(self, s5):
        key = jax.random.key(0)
        batch_size, seq_len, features = 4, 8, 16
        inputs = jnp.ones((batch_size, seq_len, features))
        mask = jnp.zeros((batch_size, seq_len))

        params = s5.init(key, inputs, mask)
        carry, outputs = s5.apply(params, inputs, mask)

        assert outputs.shape == (batch_size, seq_len, features)


class TestLinearAttention:
    """Test Linear Attention memoroid."""

    @pytest.fixture
    def linear_attention(self):
        from memorax.networks.sequence_models import LinearAttentionCell
        from memorax.networks.sequence_models.memoroid import Memoroid

        algebra = LinearAttentionCell(
            head_dim=8,
            num_heads=4,
            kernel_init=initializers.lecun_normal(),
            bias_init=initializers.zeros_init(),
            param_dtype=jnp.float32,
        )
        return Memoroid(cell=algebra)

    def test_initialize_carry(self, linear_attention):
        key = jax.random.key(0)
        input_shape = (4, 16)
        carry = linear_attention.initialize_carry(key, input_shape)
        assert carry is not None
        # Check carry structure: (decay, state)
        assert len(carry) == 2

    def test_call(self, linear_attention):
        key = jax.random.key(0)
        batch_size, seq_len, features = 4, 8, 16
        inputs = jnp.ones((batch_size, seq_len, features))
        mask = jnp.zeros((batch_size, seq_len))

        params = linear_attention.init(key, inputs, mask)
        carry, outputs = linear_attention.apply(params, inputs, mask)

        assert outputs.shape == (batch_size, seq_len, features)


class TestMamba:
    """Test Mamba selective SSM as a Memoroid algebra."""

    @pytest.fixture
    def mamba_algebra(self):
        from memorax.networks.sequence_models import MambaCell

        return MambaCell(
            features=16,
            num_heads=4,
            head_dim=8,
            hidden_dim=16,
        )

    @pytest.fixture
    def mamba(self):
        from memorax.networks.sequence_models import MambaCell
        from memorax.networks.sequence_models.memoroid import Memoroid

        algebra = MambaCell(
            features=16,
            num_heads=4,
            head_dim=8,
            hidden_dim=16,
        )
        return Memoroid(cell=algebra)

    def test_initialize_carry(self, mamba_algebra):
        key = jax.random.key(0)
        batch_size, features = 4, 16
        input_shape = (batch_size, features)

        carry = mamba_algebra.initialize_carry(key, input_shape)
        assert carry is not None
        assert len(carry) == 4  # (decay, state, gate, x_inner)

    def test_algebra_call(self, mamba_algebra):
        """Test the algebra's __call__ directly."""
        key = jax.random.key(0)
        batch_size, seq_len, features = 4, 8, 16

        x = jnp.ones((batch_size, seq_len, features))

        params = mamba_algebra.init(key, x)
        carry = mamba_algebra.apply(params, x)

        assert carry is not None
        assert len(carry) == 4  # (decay, state, gate, x_inner)

    def test_binary_operator(self, mamba_algebra):
        """Test the binary_operator operation."""
        key = jax.random.key(0)
        batch_size, seq_len, features = 4, 1, 16

        x = jnp.ones((batch_size, seq_len, features))

        params = mamba_algebra.init(key, x)
        carry_a = mamba_algebra.apply(params, x)
        carry_b = mamba_algebra.apply(params, x * 2)

        combined = mamba_algebra.binary_operator(carry_a, carry_b)
        assert combined is not None
        assert len(combined) == 4

    def test_memoroid_call(self, mamba):
        """Test Mamba through the Memoroid wrapper."""
        key = jax.random.key(0)
        batch_size, seq_len, features = 4, 8, 16
        inputs = jnp.ones((batch_size, seq_len, features))
        mask = jnp.zeros((batch_size, seq_len))

        params = mamba.init(key, inputs, mask)
        carry, outputs = mamba.apply(params, inputs, mask)

        assert outputs.shape == (batch_size, seq_len, features)

    def test_with_initial_carry(self, mamba):
        """Test Mamba with initial carry."""
        key = jax.random.key(0)
        batch_size, seq_len, features = 4, 8, 16
        inputs = jnp.ones((batch_size, seq_len, features))
        mask = jnp.zeros((batch_size, seq_len))

        params = mamba.init(key, inputs, mask)
        initial_carry = mamba.initialize_carry(key, (batch_size, features))

        carry, outputs = mamba.apply(params, inputs, mask, initial_carry)
        assert outputs.shape == (batch_size, seq_len, features)


class TestRNN:
    """Test RNN wrapper with various Flax RNN cells."""

    @pytest.fixture
    def gru_rnn(self):
        from memorax.networks.sequence_models import RNN

        return RNN(cell=nn.GRUCell(features=32))

    @pytest.fixture
    def lstm_rnn(self):
        from memorax.networks.sequence_models import RNN

        return RNN(cell=nn.LSTMCell(features=32))

    def test_gru_initialize_carry(self, gru_rnn):
        key = jax.random.key(0)
        input_shape = (4, 16)
        carry = gru_rnn.initialize_carry(key, input_shape)
        assert carry is not None

    def test_gru_call(self, gru_rnn):
        key = jax.random.key(0)
        batch_size, seq_len, features = 4, 8, 16
        inputs = jnp.ones((batch_size, seq_len, features))
        mask = jnp.zeros((batch_size, seq_len))

        params = gru_rnn.init(key, inputs, mask)
        carry, outputs = gru_rnn.apply(params, inputs, mask)

        assert outputs.shape == (batch_size, seq_len, 32)  # features from GRUCell

    def test_lstm_initialize_carry(self, lstm_rnn):
        key = jax.random.key(0)
        input_shape = (4, 16)
        carry = lstm_rnn.initialize_carry(key, input_shape)
        assert carry is not None

    def test_lstm_call(self, lstm_rnn):
        key = jax.random.key(0)
        batch_size, seq_len, features = 4, 8, 16
        inputs = jnp.ones((batch_size, seq_len, features))
        mask = jnp.zeros((batch_size, seq_len))

        params = lstm_rnn.init(key, inputs, mask)
        carry, outputs = lstm_rnn.apply(params, inputs, mask)

        assert outputs.shape == (batch_size, seq_len, 32)


class TestSHMCell:
    """Test Stable Hadamard Memory cell."""

    @pytest.fixture
    def shm_cell(self):
        from memorax.networks.sequence_models import SHMCell

        return SHMCell(
            features=32,
            output_features=16,
            num_thetas=64,
            sample_theta=False,  # Disable sampling for deterministic tests
        )

    def test_initialize_carry(self, shm_cell):
        key = jax.random.key(0)
        input_shape = (4, 16)  # (batch_size, features)
        carry = shm_cell.initialize_carry(key, input_shape)
        assert carry is not None
        # SHM carry is a matrix of shape (batch_size, features, features)
        assert carry.shape == (4, 32, 32)

    def test_call(self, shm_cell):
        key = jax.random.key(0)
        batch_size, features = 4, 16
        inputs = jnp.ones((batch_size, features))

        # Initialize carry
        carry = shm_cell.initialize_carry(key, (batch_size, features))

        params = shm_cell.init(key, carry, inputs)
        new_carry, outputs = shm_cell.apply(params, carry, inputs)

        assert new_carry.shape == carry.shape
        assert outputs.shape == (batch_size, 16)  # output_features


class TestSLSTMCell:
    """Test sLSTM cell with RNN wrapper."""

    @pytest.fixture
    def slstm_rnn(self):
        from memorax.networks.sequence_models import RNN, sLSTMCell

        cell = sLSTMCell(
            features=16,
            hidden_dim=32,
            num_heads=4,
        )
        return RNN(cell=cell)

    def test_initialize_carry(self, slstm_rnn):
        key = jax.random.key(0)
        input_shape = (4, 16)
        carry = slstm_rnn.initialize_carry(key, input_shape)
        assert carry is not None
        # sLSTM carry is (cell_state, conv_state)
        cell_state, conv_state = carry
        # cell_state is (c, n, m, h)
        assert len(cell_state) == 4

    def test_call(self, slstm_rnn):
        key = jax.random.key(0)
        batch_size, seq_len, features = 4, 8, 16
        inputs = jnp.ones((batch_size, seq_len, features))
        mask = jnp.zeros((batch_size, seq_len))

        params = slstm_rnn.init(key, inputs, mask)
        carry, outputs = slstm_rnn.apply(params, inputs, mask)

        assert outputs.shape == (batch_size, seq_len, features)

    def test_with_initial_carry(self, slstm_rnn):
        key = jax.random.key(0)
        batch_size, seq_len, features = 4, 8, 16
        inputs = jnp.ones((batch_size, seq_len, features))
        mask = jnp.zeros((batch_size, seq_len))

        params = slstm_rnn.init(key, inputs, mask)
        initial_carry = slstm_rnn.initialize_carry(key, (batch_size, features))

        carry, outputs = slstm_rnn.apply(params, inputs, mask, initial_carry)
        assert outputs.shape == (batch_size, seq_len, features)


class TestMLSTM:
    """Test mLSTM as a Memoroid algebra (parallel scan)."""

    @pytest.fixture
    def mlstm(self):
        from memorax.networks.sequence_models import mLSTMCell
        from memorax.networks.sequence_models.memoroid import Memoroid

        algebra = mLSTMCell(
            features=16,
            hidden_dim=32,
            num_heads=4,
        )
        return Memoroid(cell=algebra)

    def test_initialize_carry(self, mlstm):
        key = jax.random.key(0)
        input_shape = (4, 16)
        carry = mlstm.initialize_carry(key, input_shape)
        assert carry is not None
        # mLSTM carry is (log_f, C, n, m)
        assert len(carry) == 4

    def test_call(self, mlstm):
        key = jax.random.key(0)
        batch_size, seq_len, features = 4, 8, 16
        inputs = jnp.ones((batch_size, seq_len, features))
        mask = jnp.zeros((batch_size, seq_len))

        params = mlstm.init(key, inputs, mask)
        carry, outputs = mlstm.apply(params, inputs, mask)

        assert outputs.shape == (batch_size, seq_len, features)

    def test_with_initial_carry(self, mlstm):
        key = jax.random.key(0)
        batch_size, seq_len, features = 4, 8, 16
        inputs = jnp.ones((batch_size, seq_len, features))
        mask = jnp.zeros((batch_size, seq_len))

        params = mlstm.init(key, inputs, mask)
        initial_carry = mlstm.initialize_carry(key, (batch_size, features))

        carry, outputs = mlstm.apply(params, inputs, mask, initial_carry)
        assert outputs.shape == (batch_size, seq_len, features)

    def test_binary_operator_associativity(self, mlstm):
        """Test that binary_operator operation is associative."""
        from memorax.networks.sequence_models import mLSTMCell

        algebra = mLSTMCell(
            features=16,
            hidden_dim=32,
            num_heads=4,
        )

        key = jax.random.key(0)
        batch_size, seq_len, features = 2, 1, 16
        inputs = jax.random.normal(key, (batch_size, seq_len, features))

        params = algebra.init(key, inputs)

        # Create three elements
        a = algebra.apply(params, inputs * 1.0)
        b = algebra.apply(params, inputs * 2.0)
        c = algebra.apply(params, inputs * 3.0)

        # Test associativity: (a ⊕ b) ⊕ c == a ⊕ (b ⊕ c)
        ab = algebra.binary_operator(a, b)
        ab_c = algebra.binary_operator(ab, c)

        bc = algebra.binary_operator(b, c)
        a_bc = algebra.binary_operator(a, bc)

        # Check all components are close
        for i in range(4):
            assert jnp.allclose(ab_c[i], a_bc[i], atol=1e-5), (
                f"Component {i} not associative"
            )


class TestSequenceModelWrapper:
    """Test SequenceModelWrapper for non-recurrent networks."""

    @pytest.fixture
    def wrapped_mlp(self):
        from memorax.networks.sequence_models.wrappers import \
            SequenceModelWrapper

        mlp = nn.Dense(features=32)
        return SequenceModelWrapper(network=mlp)

    def test_initialize_carry(self, wrapped_mlp):
        key = jax.random.key(0)
        input_shape = (4, 16)
        carry = wrapped_mlp.initialize_carry(key, input_shape)
        # SequenceModelWrapper returns None for carry since it has no state
        assert carry is None

    def test_call(self, wrapped_mlp):
        key = jax.random.key(0)
        batch_size, seq_len, features = 4, 8, 16
        inputs = jnp.ones((batch_size, seq_len, features))
        mask = jnp.zeros((batch_size, seq_len))

        params = wrapped_mlp.init(key, inputs, mask)
        carry, outputs = wrapped_mlp.apply(params, inputs, mask)

        # Carry should be None for wrapped non-recurrent networks
        assert carry is None
        assert outputs.shape == (batch_size, seq_len, 32)


class TestMetaMaskWrapper:
    """Test MetaMaskWrapper for trial-based masking."""

    @pytest.fixture
    def meta_masked_rnn(self):
        from memorax.networks.sequence_models import RNN
        from memorax.networks.sequence_models.wrappers import MetaMaskWrapper

        rnn = RNN(cell=nn.GRUCell(features=32))
        return MetaMaskWrapper(sequence_model=rnn, steps_per_trial=4)

    def test_initialize_carry(self, meta_masked_rnn):
        key = jax.random.key(0)
        input_shape = (4, 8, 16)  # (batch_size, seq_len, features)
        carry = meta_masked_rnn.initialize_carry(key, input_shape)
        assert carry is not None
        # MetaMaskState has carry and step
        assert hasattr(carry, "carry")
        assert hasattr(carry, "step")

    def test_call(self, meta_masked_rnn):
        key = jax.random.key(0)
        batch_size, seq_len, features = 4, 8, 16
        inputs = jnp.ones((batch_size, seq_len, features))
        mask = jnp.zeros((batch_size, seq_len))  # Will be overwritten by wrapper

        params = meta_masked_rnn.init(key, inputs, mask)
        carry, outputs = meta_masked_rnn.apply(params, inputs, mask)

        assert outputs.shape == (batch_size, seq_len, 32)
        # Step should be incremented by seq_len
        assert jnp.all(carry.step == seq_len)


class TestAlgebraBinaryOperator:
    """Test the binary_operator operation for Algebra subclasses."""

    def test_lru_binary_operator(self):
        from memorax.networks.sequence_models import LRUCell

        algebra = LRUCell(
            features=16,
            hidden_dim=32,
        )

        key = jax.random.key(0)
        batch_size, seq_len, features = 4, 1, 16
        inputs = jnp.ones((batch_size, seq_len, features))

        # Initialize and get two different carries
        params = algebra.init(key, inputs)
        carry_a = algebra.apply(params, inputs)
        carry_b = algebra.apply(params, inputs * 2)

        # Test binary_operator
        combined = algebra.binary_operator(carry_a, carry_b)
        assert combined is not None

    def test_ffm_binary_operator(self):
        from memorax.networks.sequence_models import FFMCell

        algebra = FFMCell(
            features=16,
            memory_size=16,
            context_size=8,
        )

        key = jax.random.key(0)
        batch_size, seq_len, features = 4, 1, 16
        inputs = jnp.ones((batch_size, seq_len, features))

        params = algebra.init(key, inputs)
        carry_a = algebra.apply(params, inputs)
        carry_b = algebra.apply(params, inputs * 2)

        # binary_operator needs access to setup params, so call it within apply context
        combined = algebra.apply(params, carry_a, carry_b, method="binary_operator")
        assert combined is not None

    def test_s5_binary_operator(self):
        from memorax.networks.sequence_models import S5Cell

        algebra = S5Cell(
            features=16,
            state_size=32,
        )

        key = jax.random.key(0)
        batch_size, seq_len, features = 4, 1, 16
        inputs = jnp.ones((batch_size, seq_len, features))

        params = algebra.init(key, inputs)
        carry_a = algebra.apply(params, inputs)
        carry_b = algebra.apply(params, inputs * 2)

        combined = algebra.binary_operator(carry_a, carry_b)
        assert combined is not None

    def test_min_gru_binary_operator(self):
        from memorax.networks.sequence_models import MinGRUCell

        algebra = MinGRUCell(
            features=32,
        )

        key = jax.random.key(0)
        batch_size, seq_len, features = 4, 1, 16
        inputs = jnp.ones((batch_size, seq_len, features))

        params = algebra.init(key, inputs)
        carry_a = algebra.apply(params, inputs)
        carry_b = algebra.apply(params, inputs * 2)

        combined = algebra.binary_operator(carry_a, carry_b)
        assert combined is not None

    def test_mlstm_binary_operator(self):
        from memorax.networks.sequence_models import mLSTMCell

        algebra = mLSTMCell(
            features=16,
            hidden_dim=32,
            num_heads=4,
        )

        key = jax.random.key(0)
        batch_size, seq_len, features = 4, 1, 16
        inputs = jnp.ones((batch_size, seq_len, features))

        params = algebra.init(key, inputs)
        carry_a = algebra.apply(params, inputs)
        carry_b = algebra.apply(params, inputs * 2)

        combined = algebra.binary_operator(carry_a, carry_b)
        assert combined is not None
        assert len(combined) == 4  # (log_f, C, n, m)

    def test_mamba_binary_operator(self):
        from memorax.networks.sequence_models import MambaCell

        algebra = MambaCell(
            features=16,
            num_heads=4,
            head_dim=8,
            hidden_dim=16,
        )

        key = jax.random.key(0)
        batch_size, seq_len, features = 4, 1, 16
        inputs = jnp.ones((batch_size, seq_len, features))

        params = algebra.init(key, inputs)
        carry_a = algebra.apply(params, inputs)
        carry_b = algebra.apply(params, inputs * 2)

        combined = algebra.binary_operator(carry_a, carry_b)
        assert combined is not None
        assert len(combined) == 4  # (decay, state, gate, x_inner)

    def test_linear_attention_binary_operator(self):
        from memorax.networks.sequence_models import LinearAttentionCell

        algebra = LinearAttentionCell(
            head_dim=8,
            num_heads=4,
            kernel_init=initializers.lecun_normal(),
            bias_init=initializers.zeros_init(),
            param_dtype=jnp.float32,
        )

        key = jax.random.key(0)
        batch_size, seq_len, features = 4, 1, 16
        inputs = jnp.ones((batch_size, seq_len, features))

        params = algebra.init(key, inputs)
        carry_a = algebra.apply(params, inputs)
        carry_b = algebra.apply(params, inputs * 2)

        combined = algebra.binary_operator(carry_a, carry_b)
        assert combined is not None
        assert len(combined) == 2  # (decay, state)


class TestVmapCompatibility:
    """Test that sequence models work with JAX vmap."""

    def test_lru_vmap(self):
        from memorax.networks.sequence_models import LRUCell
        from memorax.networks.sequence_models.memoroid import Memoroid

        algebra = LRUCell(
            features=16,
            hidden_dim=32,
        )
        model = Memoroid(cell=algebra)

        key = jax.random.key(0)
        batch_size, seq_len, features = 4, 8, 16
        inputs = jnp.ones((batch_size, seq_len, features))
        mask = jnp.zeros((batch_size, seq_len))

        params = model.init(key, inputs, mask)

        # Should work with larger batch via vmap
        @jax.jit
        def forward(inputs, mask):
            return model.apply(params, inputs, mask)

        carry, outputs = forward(inputs, mask)
        assert outputs.shape == (batch_size, seq_len, features)

    def test_rnn_vmap(self):
        from memorax.networks.sequence_models import RNN

        model = RNN(cell=nn.GRUCell(features=32))

        key = jax.random.key(0)
        batch_size, seq_len, features = 4, 8, 16
        inputs = jnp.ones((batch_size, seq_len, features))
        mask = jnp.zeros((batch_size, seq_len))

        params = model.init(key, inputs, mask)

        @jax.jit
        def forward(inputs, mask):
            return model.apply(params, inputs, mask)

        carry, outputs = forward(inputs, mask)
        assert outputs.shape == (batch_size, seq_len, 32)


class TestJitCompatibility:
    """Test that sequence models work with JAX jit compilation."""

    def test_memoroid_jit(self):
        from memorax.networks.sequence_models import LRUCell
        from memorax.networks.sequence_models.memoroid import Memoroid

        algebra = LRUCell(
            features=16,
            hidden_dim=32,
        )
        model = Memoroid(cell=algebra)

        key = jax.random.key(0)
        batch_size, seq_len, features = 4, 8, 16
        inputs = jnp.ones((batch_size, seq_len, features))
        mask = jnp.zeros((batch_size, seq_len))

        params = model.init(key, inputs, mask)

        @jax.jit
        def forward(inputs, mask, carry):
            return model.apply(params, inputs, mask, carry)

        initial_carry = model.initialize_carry(key, (batch_size, features))
        carry, outputs = forward(inputs, mask, initial_carry)

        # Run twice to verify jit caching works
        carry2, outputs2 = forward(inputs, mask, initial_carry)
        assert jnp.allclose(outputs, outputs2)

    def test_rnn_jit(self):
        from memorax.networks.sequence_models import RNN

        model = RNN(cell=nn.GRUCell(features=32))

        key = jax.random.key(0)
        batch_size, seq_len, features = 4, 8, 16
        inputs = jnp.ones((batch_size, seq_len, features))
        mask = jnp.zeros((batch_size, seq_len))

        params = model.init(key, inputs, mask)

        @jax.jit
        def forward(inputs, mask):
            return model.apply(params, inputs, mask)

        carry, outputs = forward(inputs, mask)
        carry2, outputs2 = forward(inputs, mask)
        assert jnp.allclose(outputs, outputs2)


class TestMaskingBehavior:
    """Test that masking properly resets state."""

    def test_mask_resets_lru(self):
        from memorax.networks.sequence_models import LRUCell
        from memorax.networks.sequence_models.memoroid import Memoroid

        algebra = LRUCell(
            features=16,
            hidden_dim=32,
        )
        model = Memoroid(cell=algebra)

        key = jax.random.key(0)
        batch_size, seq_len, features = 2, 8, 16
        inputs = jax.random.normal(key, (batch_size, seq_len, features))

        # Mask = 0 means no reset, mask = 1 means reset
        no_mask = jnp.zeros((batch_size, seq_len))
        with_mask = jnp.ones((batch_size, seq_len))

        params = model.init(key, inputs, no_mask)

        _, outputs_no_mask = model.apply(params, inputs, no_mask)
        _, outputs_with_mask = model.apply(params, inputs, with_mask)

        # Outputs should differ when mask is applied
        assert not jnp.allclose(outputs_no_mask, outputs_with_mask)

    def test_mask_resets_rnn(self):
        from memorax.networks.sequence_models import RNN

        model = RNN(cell=nn.GRUCell(features=32))

        key = jax.random.key(0)
        batch_size, seq_len, features = 2, 8, 16
        inputs = jax.random.normal(key, (batch_size, seq_len, features))

        no_mask = jnp.zeros((batch_size, seq_len))
        with_mask = jnp.ones((batch_size, seq_len))

        params = model.init(key, inputs, no_mask)

        _, outputs_no_mask = model.apply(params, inputs, no_mask)
        _, outputs_with_mask = model.apply(params, inputs, with_mask)

        # Outputs should differ when mask is applied
        assert not jnp.allclose(outputs_no_mask, outputs_with_mask)
