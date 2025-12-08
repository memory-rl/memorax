import flax.linen as nn
import gymnax  # type: ignore
import jax
import optax
import flashbax as fbx
import pytest

from memorax.algorithms.sacd import SACD, SACDConfig
from memorax.networks import heads, SeparateFeatureExtractor, Network
from memorax.networks.mlp import MLP


@pytest.fixture
def sacd_components():
    env, env_params = gymnax.make("CartPole-v1")
    action_dim = env.action_space(env_params).n

    feature_extractor = SeparateFeatureExtractor(
        observation_extractor=MLP(features=(32,))
    )
    torso = MLP(features=(32,))

    actor_network = Network(
        feature_extractor=feature_extractor,
        torso=torso,
        head=heads.Categorical(action_dim=action_dim),
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
        head=heads.DiscreteQNetwork(action_dim=action_dim),
    )
    alpha_network = heads.Alpha(initial_alpha=0.1)

    cfg = SACDConfig(
        name="test-sacd",
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
        max_grad_norm=1.0,
    )

    buffer = fbx.make_flat_buffer(
        add_batch_size=2,
        sample_batch_size=2,
        min_length=2,
        max_length=32,
        add_sequences=False,
    )

    agent = SACD(
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


def test_sacd_init(sacd_components):
    agent = sacd_components
    key = jax.random.key(0)
    agent.init(key)


def test_sacd_warmup(sacd_components):
    agent = sacd_components
    key, state = agent.init(jax.random.key(1))
    warmup_steps = agent.cfg.num_envs * 2
    agent.warmup(key, state, num_steps=warmup_steps)


def test_sacd_train(sacd_components):
    agent = sacd_components
    key, state = agent.init(jax.random.key(2))
    key, state = agent.warmup(key, state, num_steps=agent.cfg.train_frequency * 2)
    _, state, _ = agent.train(key, state, num_steps=agent.cfg.train_frequency * 2)
    assert state.step > 0


def test_sacd_evaluate(sacd_components):
    agent = sacd_components
    key, state = agent.init(jax.random.key(3))
    _, transitions = agent.evaluate(key, state, num_steps=3)
    assert transitions.action is not None
    assert transitions.action.shape[1] == agent.cfg.num_eval_envs
