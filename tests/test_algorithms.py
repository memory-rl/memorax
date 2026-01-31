import flashbax as fbx
import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
import pytest

from memorax.algorithms import (DQN, PPO, PQN, R2D2, SAC, DQNConfig, DQNState,
                                PPOConfig, PPOState, PQNConfig, PQNState,
                                R2D2Config, R2D2State, SACConfig, SACState)
from memorax.algorithms.r2d2 import compute_n_step_returns
from memorax.buffers import make_prioritised_episode_buffer
from memorax.networks import MLP, Identity, Network, SequenceModelWrapper
from memorax.networks.heads import (Alpha, Categorical, ContinuousQNetwork,
                                    DiscreteQNetwork, Gaussian,
                                    SquashedGaussian, VNetwork)
from memorax.networks.sequence_models import RNN


def make_trajectory_buffer(buffer_size, batch_size, sequence_length, num_envs):
    """Create a trajectory buffer for off-policy algorithms."""
    return fbx.make_trajectory_buffer(
        max_length_time_axis=buffer_size,
        min_length_time_axis=batch_size,
        sample_batch_size=batch_size,
        sample_sequence_length=sequence_length,
        add_batch_size=num_envs,
        period=1,
    )


class DualCriticNetwork(nn.Module):
    """Dual critic network that returns (carry, ((q1, aux1), (q2, aux2))) for SAC-style algorithms."""

    feature_extractor: nn.Module = Identity()
    torso: nn.Module = SequenceModelWrapper(Identity())
    head1: nn.Module = Identity()
    head2: nn.Module = Identity()

    @nn.compact
    def __call__(
        self,
        observation,
        mask,
        action,
        reward,
        done,
        initial_carry=None,
        **kwargs,
    ):
        x = self.feature_extractor(observation, action=action, reward=reward, done=done)
        carry, x = self.torso(
            x,
            mask=mask,
            action=action,
            reward=reward,
            done=done,
            initial_carry=initial_carry,
        )

        q1, aux1 = self.head1(x, action=action, reward=reward, done=done)
        q2, aux2 = self.head2(x, action=action, reward=reward, done=done)

        return carry, ((q1, aux1), (q2, aux2))

    @nn.nowrap
    def initialize_carry(self, input_shape):
        key = jax.random.key(0)
        carry = None
        if self.torso is not None:
            carry = self.torso.initialize_carry(key, input_shape)
        return carry


class TestDQN:
    """Test DQN algorithm with discrete action space (CartPole)."""

    @pytest.fixture
    def agent(self, cartpole_env):
        env, env_params = cartpole_env
        action_dim = env.action_space(env_params).n

        cfg = DQNConfig(
            name="test_dqn",
            learning_rate=1e-3,
            num_envs=2,
            num_eval_envs=2,
            buffer_size=256,
            gamma=0.99,
            tau=1.0,
            target_network_frequency=10,
            batch_size=16,
            start_e=1.0,
            end_e=0.01,
            exploration_fraction=0.1,
            learning_starts=32,
            train_frequency=4,
        )

        q_network = Network(
            feature_extractor=MLP(features=32),
            torso=SequenceModelWrapper(Identity()),
            head=DiscreteQNetwork(action_dim=action_dim),
        )

        optimizer = optax.adam(cfg.learning_rate)
        epsilon_schedule = optax.linear_schedule(
            init_value=cfg.start_e,
            end_value=cfg.end_e,
            transition_steps=int(cfg.exploration_fraction * 1000),
        )

        buffer = make_trajectory_buffer(
            buffer_size=cfg.buffer_size,
            batch_size=cfg.batch_size,
            sequence_length=1,
            num_envs=cfg.num_envs,
        )

        return DQN(
            cfg=cfg,
            env=env,
            env_params=env_params,
            q_network=q_network,
            optimizer=optimizer,
            buffer=buffer,
            epsilon_schedule=epsilon_schedule,
        )

    def test_init(self, agent, random_key):
        key, state = agent.init(random_key)
        assert isinstance(state, DQNState)
        assert state.step == 0
        assert isinstance(key, jax.Array)

    def test_warmup(self, agent, random_key):
        key, state = agent.init(random_key)
        key, state = agent.warmup(key, state, num_steps=32)
        assert state.step > 0

    def test_train(self, agent, random_key):
        key, state = agent.init(random_key)
        key, state = agent.warmup(key, state, num_steps=64)
        key, state, transitions = agent.train(key, state, num_steps=64)
        assert transitions is not None

    def test_evaluate(self, agent, random_key):
        key, state = agent.init(random_key)
        key, transitions = agent.evaluate(key, state, num_steps=32)
        assert transitions is not None

    def test_batch_major_train(self, agent, random_key):
        """Train should return batch-major transitions (batch, time, ...)."""
        key, state = agent.init(random_key)
        key, state = agent.warmup(key, state, num_steps=64)
        key, state, transitions = agent.train(key, state, num_steps=64)

        assert transitions.reward.shape[0] > transitions.reward.shape[1]

    def test_time_major_evaluate(self, agent, random_key):
        """Evaluate should return time-major transitions (time, batch, ...)."""
        key, state = agent.init(random_key)
        num_steps = 32
        key, transitions = agent.evaluate(key, state, num_steps=num_steps)

        assert transitions.reward.shape[0] == num_steps
        assert transitions.reward.shape[1] == agent.cfg.num_eval_envs


class TestPPODiscrete:
    """Test PPO algorithm with discrete action space (CartPole)."""

    @pytest.fixture
    def agent(self, cartpole_env):
        env, env_params = cartpole_env
        action_dim = env.action_space(env_params).n

        cfg = PPOConfig(
            name="test_ppo_discrete",
            num_envs=2,
            num_eval_envs=2,
            num_steps=16,
            gamma=0.99,
            gae_lambda=0.95,
            num_minibatches=2,
            update_epochs=2,
            normalize_advantage=True,
            clip_coef=0.2,
            clip_vloss=True,
            ent_coef=0.01,
            vf_coef=0.5,
        )

        actor = Network(
            feature_extractor=MLP(features=32),
            torso=SequenceModelWrapper(Identity()),
            head=Categorical(action_dim=action_dim),
        )

        critic = Network(
            feature_extractor=MLP(features=32),
            torso=SequenceModelWrapper(Identity()),
            head=VNetwork(),
        )

        actor_optimizer = optax.adam(1e-3)
        critic_optimizer = optax.adam(1e-3)

        return PPO(
            cfg=cfg,
            env=env,
            env_params=env_params,
            actor_network=actor,
            critic_network=critic,
            actor_optimizer=actor_optimizer,
            critic_optimizer=critic_optimizer,
        )

    def test_init(self, agent, random_key):
        key, state = agent.init(random_key)
        assert isinstance(state, PPOState)
        assert state.step == 0
        assert isinstance(key, jax.Array)

    def test_warmup(self, agent, random_key):
        key, state = agent.init(random_key)
        key, state = agent.warmup(key, state, num_steps=32)

        assert state.step == 0

    def test_train(self, agent, random_key):
        key, state = agent.init(random_key)
        key, state, transitions = agent.train(key, state, num_steps=64)
        assert transitions is not None
        assert state.step > 0

    def test_evaluate(self, agent, random_key):
        key, state = agent.init(random_key)
        key, transitions = agent.evaluate(key, state, num_steps=32)
        assert transitions is not None

    def test_batch_major_train(self, agent, random_key):
        """Train should return batch-major transitions (batch, time, ...)."""
        key, state = agent.init(random_key)
        key, state, transitions = agent.train(key, state, num_steps=64)

        assert transitions.reward.shape[1] == agent.cfg.num_steps

    def test_time_major_evaluate(self, agent, random_key):
        """Evaluate should return time-major transitions (time, batch, ...)."""
        key, state = agent.init(random_key)
        num_steps = 32
        key, transitions = agent.evaluate(key, state, num_steps=num_steps)

        assert transitions.reward.shape[0] == num_steps
        assert transitions.reward.shape[1] == agent.cfg.num_eval_envs


class TestPPOContinuous:
    """Test PPO algorithm with continuous action space (Pendulum)."""

    @pytest.fixture
    def agent(self, pendulum_env):
        env, env_params = pendulum_env
        action_dim = env.action_space(env_params).shape[0]

        cfg = PPOConfig(
            name="test_ppo_continuous",
            num_envs=2,
            num_eval_envs=2,
            num_steps=16,
            gamma=0.99,
            gae_lambda=0.95,
            num_minibatches=2,
            update_epochs=2,
            normalize_advantage=True,
            clip_coef=0.2,
            clip_vloss=True,
            ent_coef=0.01,
            vf_coef=0.5,
        )

        actor = Network(
            feature_extractor=MLP(features=32),
            torso=SequenceModelWrapper(Identity()),
            head=Gaussian(action_dim=action_dim),
        )

        critic = Network(
            feature_extractor=MLP(features=32),
            torso=SequenceModelWrapper(Identity()),
            head=VNetwork(),
        )

        actor_optimizer = optax.adam(1e-3)
        critic_optimizer = optax.adam(1e-3)

        return PPO(
            cfg=cfg,
            env=env,
            env_params=env_params,
            actor_network=actor,
            critic_network=critic,
            actor_optimizer=actor_optimizer,
            critic_optimizer=critic_optimizer,
        )

    def test_init(self, agent, random_key):
        key, state = agent.init(random_key)
        assert isinstance(state, PPOState)
        assert state.step == 0
        assert isinstance(key, jax.Array)

    def test_warmup(self, agent, random_key):
        key, state = agent.init(random_key)
        key, state = agent.warmup(key, state, num_steps=32)

        assert state.step == 0

    def test_train(self, agent, random_key):
        key, state = agent.init(random_key)
        key, state, transitions = agent.train(key, state, num_steps=64)
        assert transitions is not None
        assert state.step > 0

    def test_evaluate(self, agent, random_key):
        key, state = agent.init(random_key)
        key, transitions = agent.evaluate(key, state, num_steps=32)
        assert transitions is not None

    def test_batch_major_train(self, agent, random_key):
        """Train should return batch-major transitions (batch, time, ...)."""
        key, state = agent.init(random_key)
        key, state, transitions = agent.train(key, state, num_steps=64)

        assert transitions.reward.shape[1] == agent.cfg.num_steps

    def test_time_major_evaluate(self, agent, random_key):
        """Evaluate should return time-major transitions (time, batch, ...)."""
        key, state = agent.init(random_key)
        num_steps = 32
        key, transitions = agent.evaluate(key, state, num_steps=num_steps)

        assert transitions.reward.shape[0] == num_steps
        assert transitions.reward.shape[1] == agent.cfg.num_eval_envs


class TestPQN:
    """Test PQN algorithm with discrete action space (CartPole)."""

    @pytest.fixture
    def agent(self, cartpole_env):
        env, env_params = cartpole_env
        action_dim = env.action_space(env_params).n

        cfg = PQNConfig(
            name="test_pqn",
            num_envs=2,
            num_eval_envs=2,
            num_steps=16,
            gamma=0.99,
            td_lambda=0.95,
            num_minibatches=2,
            update_epochs=2,
        )

        q_network = Network(
            feature_extractor=MLP(features=32),
            torso=SequenceModelWrapper(Identity()),
            head=DiscreteQNetwork(action_dim=action_dim),
        )

        optimizer = optax.adam(1e-3)
        epsilon_schedule = optax.linear_schedule(
            init_value=1.0,
            end_value=0.01,
            transition_steps=1000,
        )

        return PQN(
            cfg=cfg,
            env=env,
            env_params=env_params,
            q_network=q_network,
            optimizer=optimizer,
            epsilon_schedule=epsilon_schedule,
        )

    def test_init(self, agent, random_key):
        key, state = agent.init(random_key)
        assert isinstance(state, PQNState)
        assert state.step == 0
        assert isinstance(key, jax.Array)

    def test_warmup(self, agent, random_key):
        key, state = agent.init(random_key)
        key, state = agent.warmup(key, state, num_steps=32)

        assert state.step == 0

    def test_train(self, agent, random_key):
        key, state = agent.init(random_key)
        key, state, transitions = agent.train(key, state, num_steps=64)
        assert transitions is not None
        assert state.step > 0

    def test_evaluate(self, agent, random_key):
        key, state = agent.init(random_key)
        key, transitions = agent.evaluate(key, state, num_steps=32)
        assert transitions is not None

    def test_batch_major_train(self, agent, random_key):
        """Train should return batch-major transitions (batch, time, ...)."""
        key, state = agent.init(random_key)
        key, state, transitions = agent.train(key, state, num_steps=64)

        assert transitions.reward.shape[1] == agent.cfg.num_steps

    def test_time_major_evaluate(self, agent, random_key):
        """Evaluate should return time-major transitions (time, batch, ...)."""
        key, state = agent.init(random_key)
        num_steps = 32
        key, transitions = agent.evaluate(key, state, num_steps=num_steps)

        assert transitions.reward.shape[0] == num_steps
        assert transitions.reward.shape[1] == agent.cfg.num_eval_envs


class TestSAC:
    """Test SAC algorithm with continuous action space (Pendulum)."""

    @pytest.fixture
    def agent(self, pendulum_env):
        env, env_params = pendulum_env
        action_dim = env.action_space(env_params).shape[0]

        cfg = SACConfig(
            name="test_sac",
            actor_lr=1e-3,
            critic_lr=1e-3,
            alpha_lr=1e-3,
            num_envs=2,
            num_eval_envs=2,
            buffer_size=256,
            gamma=0.99,
            tau=0.005,
            train_frequency=4,
            target_update_frequency=1,
            batch_size=16,
            initial_alpha=1.0,
            target_entropy_scale=1.0,
            learning_starts=32,
            max_grad_norm=0.5,
        )

        actor_network = Network(
            feature_extractor=MLP(features=32),
            torso=SequenceModelWrapper(Identity()),
            head=SquashedGaussian(action_dim=action_dim),
        )

        critic_network = DualCriticNetwork(
            feature_extractor=MLP(features=32),
            torso=SequenceModelWrapper(Identity()),
            head1=ContinuousQNetwork(),
            head2=ContinuousQNetwork(),
        )

        alpha_network = Alpha(initial_alpha=cfg.initial_alpha)

        actor_optimizer = optax.adam(cfg.actor_lr)
        critic_optimizer = optax.adam(cfg.critic_lr)
        alpha_optimizer = optax.adam(cfg.alpha_lr)

        buffer = make_trajectory_buffer(
            buffer_size=cfg.buffer_size,
            batch_size=cfg.batch_size,
            sequence_length=1,
            num_envs=cfg.num_envs,
        )

        return SAC(
            cfg=cfg,
            env=env,
            env_params=env_params,
            actor_network=actor_network,
            critic_network=critic_network,
            alpha_network=alpha_network,
            actor_optimizer=actor_optimizer,
            critic_optimizer=critic_optimizer,
            alpha_optimizer=alpha_optimizer,
            buffer=buffer,
        )

    def test_init(self, agent, random_key):
        key, state = agent.init(random_key)
        assert isinstance(state, SACState)
        assert state.step == 0
        assert isinstance(key, jax.Array)

    def test_warmup(self, agent, random_key):
        key, state = agent.init(random_key)
        key, state = agent.warmup(key, state, num_steps=32)
        assert state.step > 0

    def test_train(self, agent, random_key):
        key, state = agent.init(random_key)
        key, state = agent.warmup(key, state, num_steps=64)
        key, state, transitions = agent.train(key, state, num_steps=64)
        assert transitions is not None

    def test_evaluate(self, agent, random_key):
        key, state = agent.init(random_key)
        key, transitions = agent.evaluate(key, state, num_steps=32)
        assert transitions is not None

    def test_batch_major_train(self, agent, random_key):
        """Train should return batch-major transitions (batch, time, ...)."""
        key, state = agent.init(random_key)
        key, state = agent.warmup(key, state, num_steps=64)
        key, state, transitions = agent.train(key, state, num_steps=64)

        expected_time = agent.cfg.train_frequency // agent.cfg.num_envs
        assert transitions.reward.shape[1] == expected_time

    def test_time_major_evaluate(self, agent, random_key):
        """Evaluate should return time-major transitions (time, batch, ...)."""
        key, state = agent.init(random_key)
        num_steps = 32
        key, transitions = agent.evaluate(key, state, num_steps=num_steps)

        assert transitions.reward.shape[0] == num_steps // agent.cfg.num_eval_envs
        assert transitions.reward.shape[1] == agent.cfg.num_eval_envs


class TestNStepReturns:
    """Tests for n-step return computation used by R2D2."""

    def test_n_step_returns_shape(self):
        """Output shape should be [batch, seq_len - n_step + 1]."""
        batch_size, seq_len, n_step = 4, 10, 3
        rewards = jnp.ones((batch_size, seq_len))
        dones = jnp.zeros((batch_size, seq_len))
        next_q = jnp.ones((batch_size, seq_len))

        returns = compute_n_step_returns(rewards, dones, next_q, n_step, gamma=0.99)

        expected_targets = seq_len - n_step + 1
        assert returns.shape == (batch_size, expected_targets)

    def test_n_step_returns_no_done(self):
        """Without done flags, should accumulate full n-step return."""
        rewards = jnp.array([[1.0, 1.0, 1.0, 1.0, 1.0]])
        dones = jnp.zeros((1, 5))
        next_q = jnp.array([[0.0, 0.0, 0.0, 0.0, 10.0]])
        gamma = 0.9
        n_step = 3

        returns = compute_n_step_returns(rewards, dones, next_q, n_step, gamma)


        expected_first = 1.0 + 0.9 * 1.0 + 0.81 * 1.0 + 0.729 * 0.0
        assert jnp.isclose(returns[0, 0], expected_first, atol=1e-5)

    def test_n_step_returns_with_done(self):
        """Done flag should stop accumulation and bootstrapping."""
        rewards = jnp.array([[1.0, 1.0, 1.0, 1.0, 1.0]])
        dones = jnp.array([[0.0, 1.0, 0.0, 0.0, 0.0]])
        next_q = jnp.array([[0.0, 0.0, 0.0, 0.0, 10.0]])
        gamma = 0.9
        n_step = 3

        returns = compute_n_step_returns(rewards, dones, next_q, n_step, gamma)


        expected_first = 1.0 + 0.9 * 1.0
        assert jnp.isclose(returns[0, 0], expected_first, atol=1e-5)

    def test_n_step_returns_single_step(self):
        """With n_step=1, should be standard TD target."""
        rewards = jnp.array([[1.0, 2.0, 3.0]])
        dones = jnp.zeros((1, 3))
        next_q = jnp.array([[5.0, 6.0, 7.0]])
        gamma = 0.99

        returns = compute_n_step_returns(rewards, dones, next_q, n_step=1, gamma=gamma)

        expected = rewards + gamma * next_q
        assert jnp.allclose(returns, expected, atol=1e-5)


class TestR2D2:
    """Test R2D2 algorithm with discrete action space (CartPole)."""

    @pytest.fixture
    def agent(self, cartpole_env):
        env, env_params = cartpole_env
        action_dim = env.action_space(env_params).n

        cfg = R2D2Config(
            name="test_r2d2",
            learning_rate=1e-3,
            num_envs=2,
            num_eval_envs=2,
            buffer_size=1000,
            gamma=0.99,
            tau=1.0,
            target_network_frequency=10,
            batch_size=4,
            start_e=1.0,
            end_e=0.01,
            exploration_fraction=0.1,
            learning_starts=50,
            train_frequency=4,
            burn_in_length=2,
            sequence_length=8,
            n_step=2,
            priority_exponent=0.6,
            importance_sampling_exponent=0.4,
            double=True,
        )

        q_network = Network(
            feature_extractor=MLP(features=32),
            torso=RNN(cell=nn.GRUCell(features=32)),
            head=DiscreteQNetwork(action_dim=action_dim),
        )

        optimizer = optax.adam(cfg.learning_rate)
        epsilon_schedule = optax.linear_schedule(
            init_value=cfg.start_e,
            end_value=cfg.end_e,
            transition_steps=1000,
        )
        beta_schedule = optax.linear_schedule(
            init_value=cfg.importance_sampling_exponent,
            end_value=1.0,
            transition_steps=1000,
        )

        buffer = make_prioritised_episode_buffer(
            max_length=cfg.buffer_size,
            min_length=cfg.batch_size * cfg.sequence_length,
            sample_batch_size=cfg.batch_size,
            sample_sequence_length=cfg.sequence_length,
            add_batch_size=cfg.num_envs,
            add_sequences=True,
            priority_exponent=cfg.priority_exponent,
        )

        return R2D2(
            cfg=cfg,
            env=env,
            env_params=env_params,
            q_network=q_network,
            optimizer=optimizer,
            buffer=buffer,
            epsilon_schedule=epsilon_schedule,
            beta_schedule=beta_schedule,
        )

    def test_init(self, agent, random_key):
        key, state = agent.init(random_key)
        assert isinstance(state, R2D2State)
        assert state.step == 0
        assert isinstance(key, jax.Array)

    def test_warmup(self, agent, random_key):
        key, state = agent.init(random_key)
        key, state = agent.warmup(key, state, num_steps=100)
        assert state.step > 0

    def test_train(self, agent, random_key):
        key, state = agent.init(random_key)
        key, state = agent.warmup(key, state, num_steps=200)
        key, state, transitions = agent.train(key, state, num_steps=16)
        assert transitions is not None
        assert "losses/loss" in transitions.info

    def test_evaluate(self, agent, random_key):
        key, state = agent.init(random_key)
        key, transitions = agent.evaluate(key, state, num_steps=32)
        assert transitions is not None

    def test_batch_major_train(self, agent, random_key):
        """Train should return batch-major transitions (batch, time, ...)."""
        key, state = agent.init(random_key)
        key, state = agent.warmup(key, state, num_steps=200)
        key, state, transitions = agent.train(key, state, num_steps=16)

        expected_time = agent.cfg.train_frequency // agent.cfg.num_envs
        assert transitions.reward.shape[1] == expected_time

    def test_time_major_evaluate(self, agent, random_key):
        """Evaluate should return time-major transitions (time, batch, ...)."""
        key, state = agent.init(random_key)
        num_steps = 32
        key, transitions = agent.evaluate(key, state, num_steps=num_steps)

        assert transitions.reward.shape[0] == num_steps
        assert transitions.reward.shape[1] == agent.cfg.num_eval_envs
