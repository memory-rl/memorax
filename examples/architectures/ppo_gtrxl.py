"""PPO on CartPole with GTrXL (Gated Transformer-XL) architecture.

This example demonstrates using Gated Transformer-XL with:
- Multi-head self-attention with segment-level recurrence
- Gated residual connections (learned gates)
- Memory that persists across segments for long-range dependencies
- Relative positional embeddings (ALiBi)
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
    ALiBi,
    FeatureExtractor,
    GatedResidual,
    Network,
    PreNorm,
    SegmentRecurrence,
    SelfAttention,
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
    name="PPO-GTrXL",
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

# GTrXL configuration
d_model = 64
num_heads = 4
context_length = 128
memory_length = 64
num_layers = 2

feature_extractor = FeatureExtractor(
    observation_extractor=MLP(
        features=(d_model,), kernel_init=nn.initializers.orthogonal(scale=1.414)
    ),
)


# GTrXL: SegmentRecurrence wraps attention for memory, GatedResidual for gating
attention = GatedResidual(
    module=PreNorm(
        module=SegmentRecurrence(
            module=SelfAttention(
                features=d_model,
                num_heads=num_heads,
                context_length=context_length,
                dtype=jnp.float32,
                param_dtype=jnp.float32,
                kernel_init=nn.initializers.xavier_uniform(),
                bias_init=nn.initializers.zeros_init(),
                positional_embedding=ALiBi(num_heads=num_heads),
            ),
            memory_length=memory_length,
            features=d_model,
        )
    )
)
ffn = GatedResidual(module=PreNorm(module=FFN(features=d_model, expansion_factor=4)))

torso = Stack(blocks=(attention, ffn) * num_layers)

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
    [DashboardLogger(title="PPO GTrXL CartPole", total_timesteps=total_timesteps)]
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
