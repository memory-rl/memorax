import flax.linen as nn
import gymnax  # type: ignore
import jax
import optax
import pytest

from memorax.algorithms.rpqn import RPQN, RPQNConfig
from memorax.networks import RecurrentNetwork, heads
from memorax.networks.feature_extractors import SeparateFeatureExtractor
from memorax.networks.mlp import MLP


@pytest.fixture
def rpqn_components():
    env, env_params = gymnax.make("CartPole-v1")

    feature_extractor = SeparateFeatureExtractor(
        observation_extractor=MLP(features=(16,))
    )
    torso = nn.GRUCell(features=16)
    head = heads.DiscreteQNetwork(
        action_dim=env.action_space(env_params).n,
        kernel_init=nn.initializers.orthogonal(scale=1.0),
    )

    q_network = RecurrentNetwork(
        feature_extractor=feature_extractor,
        torso=torso,
        head=head,
    )

    cfg = RPQNConfig(
        name="test-rpqn",
        learning_rate=1e-3,
        num_envs=2,
        num_eval_envs=2,
        num_steps=4,
        gamma=0.99,
        td_lambda=0.95,
        num_minibatches=2,
        update_epochs=2,
        start_e=1.0,
        end_e=0.1,
        exploration_fraction=0.5,
        max_grad_norm=0.5,
        learning_starts=0,
    )

    optimizer = optax.adam(cfg.learning_rate)
    epsilon_schedule = optax.linear_schedule(
        init_value=1.0,
        end_value=0.1,
        transition_steps=10,
        transition_begin=cfg.learning_starts,
    )

    agent = RPQN(
        cfg=cfg,
        env=env,
        env_params=env_params,
        q_network=q_network,
        optimizer=optimizer,
        epsilon_schedule=epsilon_schedule,
    )
    return agent


def test_rpqn_init(rpqn_components):
    agent = rpqn_components
    key = jax.random.key(0)
    agent.init(key)


def test_rpqn_warmup(rpqn_components):
    agent = rpqn_components
    key = jax.random.key(1)
    key, state = agent.init(key)
    warmup_steps = 8
    agent.warmup(key, state, num_steps=warmup_steps)


def test_rpqn_train(rpqn_components):
    agent = rpqn_components
    key, state = agent.init(jax.random.key(0))
    rollout_steps = agent.cfg.num_steps * agent.cfg.num_envs
    agent.train(key, state, num_steps=rollout_steps)


def test_rpqn_evaluate(rpqn_components):
    agent = rpqn_components
    key, state = agent.init(jax.random.key(0))
    agent.evaluate(key, state, num_steps=3)
