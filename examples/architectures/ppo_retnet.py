"""PPO on CartPole with RetNet (Retentive Network) architecture.

This example demonstrates using RetNet with:
- Multi-scale retention mechanism (exponential decay instead of softmax)
- Different decay rates per head for multi-temporal modeling
- Efficient parallel training via associative scan
- Linear complexity sequence modeling
"""
import time
from dataclasses import asdict

import flax.linen as nn
import jax
import jax.numpy as jnp
import optax

from memorax.algorithms import PPO, PPOConfig
from memorax.environments import environment
from memorax.loggers import DashboardLogger, Logger
from memorax.networks import (
    MLP,
    FFN,
    FeatureExtractor,
    Memoroid,
    Network,
    PreNorm,
    Residual,
    RetentionCell,
    Stack,
    heads,
)

total_timesteps = 500_000
num_train_steps = 10_000
num_eval_steps = 10_000

seed = 0
num_seeds = 1

env, env_params = environment.make("gymnax::CartPole-v1")

cfg = PPOConfig(
    name="PPO-RetNet",
    num_envs=8,
    num_eval_envs=16,
    num_steps=128,
    gamma=0.99,
    gae_lambda=0.95,
    num_minibatches=4,
    update_epochs=4,
    normalize_advantage=True,
    clip_coef=0.2,
    clip_vloss=True,
    ent_coef=0.01,
    vf_coef=0.5,
)

# RetNet configuration
d_model = 64
num_heads = 4
head_dim = 16
num_layers = 2

feature_extractor = FeatureExtractor(
    observation_extractor=MLP(
        features=(d_model,), kernel_init=nn.initializers.orthogonal(scale=1.414)
    ),
)


# RetNet block with FFN
retention = Residual(
    module=PreNorm(
        module=Memoroid(
            cell=RetentionCell(
                head_dim=head_dim,
                num_heads=num_heads,
                kernel_init=nn.initializers.xavier_uniform(),
                bias_init=nn.initializers.zeros_init(),
                param_dtype=jnp.float32,
                dtype=jnp.float32,
                gamma_min=0.9,
                gamma_max=0.99,
            )
        )
    )
)
ffn = Residual(module=PreNorm(module=FFN(features=d_model, expansion_factor=4)))

torso = Stack(blocks=(retention, ffn) * num_layers)

actor_network = Network(
    feature_extractor=feature_extractor,
    torso=torso,
    head=heads.Categorical(
        action_dim=env.action_space(env_params).n,
        kernel_init=nn.initializers.orthogonal(scale=0.01),
    ),
)

critic_network = Network(
    feature_extractor=feature_extractor,
    torso=torso,
    head=heads.VNetwork(
        kernel_init=nn.initializers.orthogonal(scale=1.0),
    ),
)

optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adam(learning_rate=3e-4, eps=1e-5),
)


key = jax.random.key(seed)
keys = jax.random.split(key, num_seeds)

agent = PPO(
    cfg=cfg,
    env=env,
    env_params=env_params,
    actor=actor_network,
    critic=critic_network,
    actor_optimizer=optimizer,
    critic_optimizer=optimizer,
)

logger = Logger(
    [DashboardLogger(title="PPO RetNet CartPole", total_timesteps=total_timesteps)]
)
logger_state = logger.init(cfg=asdict(cfg))

init = jax.vmap(agent.init)
evaluate = jax.vmap(agent.evaluate, in_axes=(0, 0, None))
train = jax.vmap(agent.train, in_axes=(0, 0, None))

keys, state = init(keys)

keys, transitions = evaluate(keys, state, num_eval_steps)
evaluation_statistics = jax.vmap(Logger.get_episode_statistics, in_axes=(0, None))(
    transitions, "evaluation"
)
logger_state = logger.log(
    logger_state, evaluation_statistics, step=state.step[0].item()
)
logger.emit(logger_state)

for i in range(0, total_timesteps, num_train_steps):
    start = time.perf_counter()
    keys, state, transitions = train(keys, state, num_train_steps)
    jax.block_until_ready(state)
    end = time.perf_counter()

    SPS = num_train_steps / (end - start)

    training_statistics = jax.vmap(Logger.get_episode_statistics, in_axes=(0, None))(
        transitions, "training"
    )
    losses = jax.vmap(
        lambda transition: jax.tree.map(lambda x: x.mean(), transition.losses)
    )(transitions)
    data = {"SPS": SPS, **training_statistics, **losses}
    logger_state = logger.log(logger_state, data, step=state.step[0].item())

    keys, transitions = evaluate(keys, state, num_eval_steps)
    evaluation_statistics = jax.vmap(Logger.get_episode_statistics, in_axes=(0, None))(
        transitions, "evaluation"
    )
    logger_state = logger.log(
        logger_state, evaluation_statistics, step=state.step[0].item()
    )
    logger.emit(logger_state)
