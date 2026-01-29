"""IPPO on MettaGrid arena environment.

This example demonstrates Independent PPO (IPPO) for multi-agent reinforcement
learning using the MettaGrid environment. Each agent learns its own policy
independently while sharing the same network architecture.

MettaGrid is a CPU-based multi-agent gridworld environment that is integrated
with JAX training via pure_callback.

Requirements:
    pip install mettagrid
"""

import os
import time
from dataclasses import asdict

import flax.linen as nn
import jax
import jax.numpy as jnp
import optax

from memorax.algorithms import IPPO, IPPOConfig
from memorax.environments.mettagrid import (
    FlattenObservationWrapper,
    MettagridEnvironment,
)
from memorax.loggers import DashboardLogger, Logger
from memorax.networks import MLP, FeatureExtractor, Network, SequenceModelWrapper, heads

total_timesteps = 100_000_000
num_train_steps = 25_000_000
num_eval_steps = 0

seed = 0
num_seeds = 1

from cogames.cogs_vs_clips.missions import make_cogsguard_mission

num_agents = 10
num_workers = 64

cfg = make_cogsguard_mission(num_agents=num_agents).make_env()
env = FlattenObservationWrapper(MettagridEnvironment(cfg, num_workers=num_workers))


cfg = IPPOConfig(
    name="IPPO",
    num_envs=1024,
    num_eval_envs=0,
    num_steps=256,
    gamma=0.99,
    gae_lambda=0.95,
    num_minibatches=8,
    update_epochs=1,
    normalize_advantage=True,
    clip_coef=0.2,
    clip_vloss=True,
    ent_coef=0.01,
    vf_coef=0.5,
)

d_model = 256

feature_extractor = FeatureExtractor(
    observation_extractor=MLP(
        features=(d_model,), kernel_init=nn.initializers.orthogonal(scale=1.414)
    ),
)
torso = SequenceModelWrapper(
    MLP(features=(d_model,), kernel_init=nn.initializers.orthogonal(scale=1.414))
)

action_space = env.action_spaces[env.agents[0]]

VmappedNetwork = nn.vmap(
    Network,
    variable_axes={"params": None},
    split_rngs={"params": False, "memory": True, "dropout": True},
    in_axes=(0, 0, 0, 0, 0, 0),
    out_axes=(0, 0),
)

actor_network = VmappedNetwork(
    feature_extractor=feature_extractor,
    torso=torso,
    head=heads.Categorical(
        action_dim=action_space.n,
        kernel_init=nn.initializers.orthogonal(scale=0.01),
    ),
)

critic_network = VmappedNetwork(
    feature_extractor=feature_extractor,
    torso=torso,
    head=heads.VNetwork(
        kernel_init=nn.initializers.orthogonal(scale=1.0),
    ),
)

optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.contrib.muon(
        learning_rate=0.005,  # Start conservative
        beta=0.95,  # Momentum decay (default)
        nesterov=True,
    ),
)

key = jax.random.key(seed)
keys = jax.random.split(key, num_seeds)

agent = IPPO(
    cfg=cfg,
    env=env,
    actor_network=actor_network,
    critic_network=critic_network,
    actor_optimizer=optimizer,
    critic_optimizer=optimizer,
)

logger = Logger(
    [DashboardLogger(title="IPPO MettaGrid Arena", total_timesteps=total_timesteps)]
)
logger_state = logger.init(cfg=asdict(cfg))

init = jax.vmap(agent.init)
evaluate = jax.vmap(agent.evaluate, in_axes=(0, 0, None))
train = jax.vmap(agent.train, in_axes=(0, 0, None))

keys, state = init(keys)

if num_eval_steps > 0:
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
    data = {"training/SPS": SPS, **training_statistics, **losses}
    logger_state = logger.log(logger_state, data, step=state.step[0].item())

    if num_eval_steps > 0:
        keys, transitions = evaluate(keys, state, num_eval_steps)
        evaluation_statistics = jax.vmap(
            Logger.get_episode_statistics, in_axes=(0, None)
        )(transitions, "evaluation")
        logger_state = logger.log(
            logger_state, evaluation_statistics, step=state.step[0].item()
        )
    logger.emit(logger_state)

env.close()
