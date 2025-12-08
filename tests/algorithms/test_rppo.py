import flax.linen as nn
import gymnax  # type: ignore
import jax
import optax
import pytest

from memorax.algorithms.rppo import RPPO, RPPOConfig
from memorax.networks import RecurrentNetwork, heads
from memorax.networks.feature_extractors import SeparateFeatureExtractor
from memorax.networks.mlp import MLP


@pytest.fixture
def rppo_components():
    env, env_params = gymnax.make("CartPole-v1")

    feature_extractor = SeparateFeatureExtractor(
        observation_extractor=MLP(features=(16,))
    )
    torso = nn.GRUCell(features=(16))

    actor = RecurrentNetwork(
        feature_extractor=feature_extractor,
        torso=torso,
        head=heads.Categorical(
            action_dim=env.action_space(env_params).n,
            kernel_init=nn.initializers.orthogonal(scale=0.01),
        ),
    )
    critic = RecurrentNetwork(
        feature_extractor=feature_extractor,
        torso=torso,
        head=heads.VNetwork(kernel_init=nn.initializers.orthogonal(scale=1.0)),
    )

    cfg = RPPOConfig(
        name="test-rppo",
        learning_rate=3e-4,
        num_envs=2,
        num_eval_envs=2,
        num_steps=16,
        anneal_lr=True,
        gamma=0.99,
        gae_lambda=0.95,
        num_minibatches=2,
        update_epochs=2,
        normalize_advantage=True,
        clip_coef=0.2,
        clip_vloss=True,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        learning_starts=0,
    )

    optimizer = optax.adam(cfg.learning_rate)

    agent = RPPO(
        cfg=cfg,
        env=env,
        env_params=env_params,
        actor=actor,
        critic=critic,
        actor_optimizer=optimizer,
        critic_optimizer=optimizer,
    )
    return agent


def test_rppo_init(rppo_components):
    agent = rppo_components
    key = jax.random.key(0)
    agent.init(key)


def test_rppo_warmup(rppo_components):
    agent = rppo_components
    key = jax.random.key(1)
    key, state = agent.init(key)
    warmup_steps = 12
    agent.warmup(key, state, num_steps=warmup_steps)


def test_rppo_train(rppo_components):
    agent = rppo_components
    key, state = agent.init(jax.random.key(0))
    agent.train(key, state, num_steps=agent.cfg.num_steps)


def test_rppo_evaluate(rppo_components):
    agent = rppo_components
    key, state = agent.init(jax.random.key(0))
    agent.evaluate(key, state, num_steps=3)
