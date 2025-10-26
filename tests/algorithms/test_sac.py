import flax.linen as nn
import gymnax  # type: ignore
import jax
import optax
import flashbax as fbx
import pytest

from memory_rl.algorithms.sac import SAC, SACConfig
from memory_rl.networks import heads, SeparateFeatureExtractor, Network
from memory_rl.networks.mlp import MLP


@pytest.fixture
def sac_components():
    env, env_params = gymnax.make("Pendulum-v1")
    action_dim = env.action_space(env_params).shape[0]

    feature_extractor = SeparateFeatureExtractor(
        observation_extractor=MLP(features=(32,))
    )
    torso = MLP(features=(32,))

    actor_network = Network(
        feature_extractor=feature_extractor,
        torso=torso,
        head=heads.SquashedGaussian(action_dim=action_dim),
    )
    critic_network = nn.vmap(
        Network,
        variable_axes={"params": 0},
        split_rngs={"params": True},
        in_axes=None,
        out_axes=0,
        axis_size=2,
    )(
        feature_extractor=feature_extractor,
        torso=torso,
        head=heads.VNetwork(),
    )
    alpha_network = heads.Alpha(initial_alpha=0.2)

    cfg = SACConfig(
        name="test-sac",
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
        learning_starts=4,
        max_grad_norm=1.0,
    )

    buffer = fbx.make_flat_buffer(
        max_length=32,
        min_length=2,
        sample_batch_size=2,
        add_sequences=False,
        add_batch_size=2,
    )

    agent = SAC(
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


def test_sac_init(sac_components):
    agent = sac_components
    key = jax.random.key(0)
    agent.init(key)


def test_sac_warmup(sac_components):
    agent = sac_components
    key = jax.random.key(1)
    key, state = agent.init(key)
    agent.warmup(key, state, num_steps=agent.cfg.learning_starts)


def test_sac_train(sac_components):
    agent = sac_components
    key = jax.random.key(2)
    key, state = agent.init(key)
    agent.train(key, state, num_steps=agent.cfg.train_frequency)


def test_sac_evaluate(sac_components):
    agent = sac_components
    key, state = agent.init(jax.random.key(0))
    agent.evaluate(key, state, num_steps=3)
