import flax.linen as nn
import gymnax  # type: ignore
import jax
import optax
import pytest

from memory_rl.algorithms.rsac import RSAC, RSACConfig
from memory_rl.buffers import make_episode_buffer
from memory_rl.networks import heads, RecurrentNetwork
from memory_rl.networks.feature_extractors import SeparateFeatureExtractor
from memory_rl.networks.mlp import MLP


@pytest.fixture
def rsac_components():
    env, env_params = gymnax.make("Pendulum-v1")
    action_dim = env.action_space(env_params).shape[0]

    feature_extractor = SeparateFeatureExtractor(
        observation_extractor=MLP(features=(32,))
    )
    torso = nn.GRUCell(features=32)

    actor_network = RecurrentNetwork(
        feature_extractor=feature_extractor,
        torso=torso,
        head=heads.SquashedGaussian(action_dim=action_dim),
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
        head=heads.VNetwork(),
    )
    alpha_network = heads.Alpha(initial_alpha=0.2)

    cfg = RSACConfig(
        name="test-rsac",
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
        initial_alpha=0.2,
        target_entropy_scale=1.0,
        learning_starts=5000,
        max_grad_norm=1.0,
        sequence_length=4,
        burn_in_length=0,
        mask=False,
        backup_entropy=False,
    )

    buffer = make_episode_buffer(
        max_length=cfg.buffer_size,
        min_length=cfg.batch_size,
        sample_batch_size=cfg.batch_size,
        sample_sequence_length=2,
        add_batch_size=cfg.num_envs,
        add_sequences=True,
    )

    agent = RSAC(
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


def test_rsac_init(rsac_components):
    agent = rsac_components
    key = jax.random.key(0)
    agent.init(key)


def test_rsac_warmup(rsac_components):
    agent = rsac_components
    key = jax.random.key(1)
    key, state = agent.init(key)
    warmup_steps = agent.cfg.num_envs * 2
    agent.warmup(key, state, num_steps=warmup_steps)


def test_rsac_train(rsac_components):
    agent = rsac_components
    key = jax.random.key(2)
    key, state = agent.init(key)
    agent.train(key, state, num_steps=agent.cfg.train_frequency)


def test_rsac_evaluate(rsac_components):
    agent = rsac_components
    key, state = agent.init(jax.random.key(3))
    agent.evaluate(key, state, num_steps=agent.cfg.num_eval_envs * 2)
