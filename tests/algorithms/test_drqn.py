import flax.linen as nn
import gymnax  # type: ignore
import jax
import flashbax as fbx
import optax
import pytest

from memorax.algorithms.drqn import DRQN, DRQNConfig
from memorax.networks import RecurrentNetwork, heads
from memorax.networks.feature_extractors import SeparateFeatureExtractor
from memorax.networks.mlp import MLP


@pytest.fixture
def drqn_components():
    env, env_params = gymnax.make("CartPole-v1")

    feature_extractor = SeparateFeatureExtractor(
        observation_extractor=MLP(features=(16,))
    )
    torso = nn.GRUCell(16)
    head = heads.DiscreteQNetwork(action_dim=env.action_space(env_params).n)

    q_network = RecurrentNetwork(
        feature_extractor=feature_extractor,
        torso=torso,
        head=head,
    )

    buffer = fbx.make_trajectory_buffer(
        add_batch_size=2,
        sample_batch_size=4,
        sample_sequence_length=4,
        min_length_time_axis=4,
        period=1,
        max_size=64,
    )

    cfg = DRQNConfig(
        name="test-drqn",
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
        sequence_length=4,
        burn_in_length=0,
        mask=False,
        double=False,
    )

    optimizer = optax.adam(cfg.learning_rate)

    epsilon_schedule = optax.linear_schedule(
        init_value=cfg.start_e,
        end_value=cfg.end_e,
        transition_steps=10,
        transition_begin=cfg.learning_starts,
    )

    agent = DRQN(
        cfg=cfg,
        env=env,
        env_params=env_params,
        q_network=q_network,
        optimizer=optimizer,
        buffer=buffer,
        epsilon_schedule=epsilon_schedule,
    )
    return agent


def test_drqn_init(drqn_components):
    agent = drqn_components
    key = jax.random.key(0)
    agent.init(key)


def test_drqn_warmup(drqn_components):
    agent = drqn_components
    key = jax.random.key(1)
    key, state = agent.init(key)
    warmup_steps = 12
    agent.warmup(key, state, num_steps=warmup_steps)


def test_drqn_train(drqn_components):
    agent = drqn_components
    key, state = agent.init(jax.random.key(0))
    agent.train(key, state, num_steps=agent.cfg.train_frequency)


def test_drqn_evaluate(drqn_components):
    agent = drqn_components
    key, state = agent.init(jax.random.key(0))
    agent.evaluate(key, state, num_steps=3)
