import flax.linen as nn
import gymnax  # type: ignore
import jax
import optax
import pytest

from memory_rl.algorithms.pqn import PQN, PQNConfig
from memory_rl.networks import Network, heads
from memory_rl.networks.feature_extractors import SeparateFeatureExtractor
from memory_rl.networks.mlp import MLP


@pytest.fixture
def pqn_components():
    env, env_params = gymnax.make("CartPole-v1")

    feature_extractor = SeparateFeatureExtractor(
        observation_extractor=MLP(features=(16,))
    )
    torso = MLP(features=(16,))
    head = heads.DiscreteQNetwork(
        action_dim=env.action_space(env_params).n,
        kernel_init=nn.initializers.orthogonal(scale=1.0),
    )

    q_network = Network(
        feature_extractor=feature_extractor,
        torso=torso,
        head=head,
    )

    cfg = PQNConfig(
        name="test-pqn",
        learning_rate=1e-3,
        num_envs=2,
        num_eval_envs=2,
        num_steps=4,
        gamma=0.99,
        td_lambda=0.95,
        num_minibatches=2,
        update_epochs=2,
        max_grad_norm=0.5,
        learning_starts=0,
        start_e=1.0,
        end_e=0.1,
        exploration_fraction=0.5,
    )

    optimizer = optax.adam(cfg.learning_rate)
    epsilon_schedule = optax.linear_schedule(
        init_value=cfg.start_e,
        end_value=cfg.end_e,
        transition_steps=10,
        transition_begin=cfg.learning_starts,
    )

    agent = PQN(
        cfg=cfg,
        env=env,
        env_params=env_params,
        q_network=q_network,
        optimizer=optimizer,
        epsilon_schedule=epsilon_schedule,
    )
    return agent


def test_pqn_init(pqn_components):
    agent = pqn_components
    key = jax.random.key(0)
    agent.init(key)


def test_pqn_warmup(pqn_components):
    agent = pqn_components
    key = jax.random.key(1)
    key, state = agent.init(key)
    warmup_steps = 8
    agent.warmup(key, state, num_steps=warmup_steps)


def test_pqn_train(pqn_components):
    agent = pqn_components
    key, state = agent.init(jax.random.key(0))
    rollout_steps = agent.cfg.num_steps * agent.cfg.num_envs
    agent.train(key, state, num_steps=rollout_steps)


def test_pqn_evaluate(pqn_components):
    agent = pqn_components
    key, state = agent.init(jax.random.key(0))
    agent.evaluate(key, state, num_steps=3)
