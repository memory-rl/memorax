import flax.linen as nn
import gymnax  # type: ignore
import jax
import flashbax as fbx
import optax
import pytest

from memory_rl.algorithms.dqn import DQN, DQNConfig
from memory_rl.networks import Network, heads
from memory_rl.networks.feature_extractors import SeparateFeatureExtractor
from memory_rl.networks.mlp import MLP


@pytest.fixture
def dqn_components():
    env, env_params = gymnax.make("CartPole-v1")

    feature_extractor = SeparateFeatureExtractor(
        observation_extractor=MLP(features=(16,))
    )
    torso = MLP(features=(16,))
    head = heads.DiscreteQNetwork(action_dim=env.action_space(env_params).n)

    q_network = Network(
        feature_extractor=feature_extractor,
        torso=torso,
        head=head,
    )

    buffer = fbx.make_flat_buffer(
        max_length=64,
        min_length=4,
        sample_batch_size=4,
        add_sequences=False,
        add_batch_size=2,
    )

    cfg = DQNConfig(
        name="test-dqn",
        learning_rate=1e-3,
        num_envs=2,
        num_eval_envs=2,
        buffer_size=64,
        gamma=0.99,
        tau=1.0,
        target_network_frequency=4,
        batch_size=4,
        start_e=1.0,
        end_e=0.1,
        exploration_fraction=0.5,
        learning_starts=0,
        train_frequency=4,
        double=False,
    )

    optimizer = optax.adam(cfg.learning_rate)

    epsilon_schedule = optax.linear_schedule(
        init_value=cfg.start_e,
        end_value=cfg.end_e,
        transition_steps=10,
        transition_begin=cfg.learning_starts,
    )

    agent = DQN(
        cfg=cfg,
        env=env,
        env_params=env_params,
        q_network=q_network,
        optimizer=optimizer,
        buffer=buffer,
        epsilon_schedule=epsilon_schedule,
    )
    return agent


def test_dqn_init(dqn_components):
    agent = dqn_components
    key = jax.random.key(0)
    agent.init(key)


def test_dqn_warmup(dqn_components):
    agent = dqn_components
    key = jax.random.key(1)
    key, state = agent.init(key)
    agent.warmup(key, state, num_steps=agent.cfg.learning_starts)


def test_dqn_train(dqn_components):
    agent = dqn_components
    key, state = agent.init(jax.random.key(0))
    agent.train(key, state, num_steps=agent.cfg.train_frequency)


def test_dqn_evaluate(dqn_components):
    agent = dqn_components
    key, state = agent.init(jax.random.key(0))
    agent.evaluate(key, state, num_steps=3)
