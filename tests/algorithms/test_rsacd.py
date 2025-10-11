from types import SimpleNamespace

import flax.linen as nn
import gymnax  # type: ignore
import jax
import jax.numpy as jnp
import optax
import pytest

from memory_rl.algorithms.rsacd import RSACD, RSACDConfig
from memory_rl.buffers import make_episode_buffer
from memory_rl.networks import RecurrentNetwork
from memory_rl.networks import heads
from memory_rl.networks.feature_extractors import SeparateFeatureExtractor
from memory_rl.networks.mlp import MLP


@pytest.fixture
def rsacd_components():
    env, env_params = gymnax.make("CartPole-v1")
    action_dim = env.action_space(env_params).n

    feature_extractor = SeparateFeatureExtractor(
        observation_extractor=MLP(features=(32,))
    )
    torso = nn.GRUCell(features=32)

    actor_network = RecurrentNetwork(
        feature_extractor=feature_extractor,
        torso=torso,
        head=heads.Categorical(action_dim=action_dim),
    )
    critic_network = nn.vmap(
        RecurrentNetwork,
        variable_axes={"params": 0},
        split_rngs={"params": True},
        in_axes=None,
        out_axes=0,
        axis_size=2,
    )(
        feature_extractor=feature_extractor,
        torso=nn.GRUCell(features=32),
        head=heads.DiscreteQNetwork(action_dim=action_dim),
    )
    alpha_network = heads.Alpha(initial_alpha=0.1)

    cfg = RSACDConfig(
        name="test-rsacd",
        actor_lr=3e-4,
        critic_lr=3e-4,
        alpha_lr=3e-4,
        num_envs=2,
        num_eval_envs=2,
        buffer_size=32,
        gamma=0.99,
        tau=0.005,
        train_frequency=2,
        target_update_frequency=2,
        batch_size=2,
        initial_alpha=0.1,
        target_entropy_scale=0.5,
        learning_starts=0,
        sequence_length=2,
        burn_in_length=0,
        mask=False,
    )

    buffer = make_episode_buffer(
        max_length=cfg.buffer_size,
        min_length=cfg.batch_size,
        sample_batch_size=cfg.batch_size,
        sample_sequence_length=1,
        add_batch_size=cfg.num_envs,
        add_sequences=True,
    )

    agent = RSACD(
        cfg=cfg,
        env=env,
        env_params=env_params,
        actor_network=actor_network,
        critic_network=critic_network,
        alpha_network=alpha_network,
        actor_optimizer=optax.adam(cfg.actor_lr),
        critic_optimizer=optax.adam(cfg.critic_lr),
        alpha_optimizer=optax.adam(cfg.alpha_lr),
        buffer=buffer,
    )
    return agent


def test_rsacd_init(rsacd_components):
    agent = rsacd_components
    key = jax.random.key(0)
    agent.init(key)


def test_rsacd_warmup(rsacd_components):
    agent = rsacd_components
    key, state = agent.init(jax.random.key(1))
    warmup_steps = agent.cfg.num_envs * 2
    agent.warmup(key, state, num_steps=warmup_steps)


def test_rsacd_train(rsacd_components):
    agent = rsacd_components
    key, state = agent.init(jax.random.key(2))
    key, state = agent.warmup(key, state, num_steps=agent.cfg.train_frequency * 2)
    _, state, _ = agent.train(key, state, num_steps=agent.cfg.train_frequency * 2)
    assert state.step > 0


def test_rsacd_evaluate(rsacd_components):
    agent = rsacd_components
    key, state = agent.init(jax.random.key(3))
    _, transitions = agent.evaluate(key, state, num_steps=agent.cfg.num_eval_envs * 2)
    assert transitions.action is not None
    assert transitions.action.shape[1] == agent.cfg.num_eval_envs
