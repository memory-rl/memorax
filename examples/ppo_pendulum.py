import time

import flax.linen as nn
import jax
import jax.numpy as jnp
import optax

from memorax.algorithms import PPO, PPOConfig
from memorax.environments import environment
from memorax.loggers import DashboardLogger, Logger, WandbLogger
from memorax.networks import (GPT2, LRU, MLP, RNN, DeltaNet, DeltaProduct,
                              FeatureExtractor, GatedDeltaNet, GTrXL, Mamba,
                              Network, TDDeltaNet, TDGatedDeltaNet, heads,
                              xLSTMCell)

total_timesteps = 1_000_000
num_train_steps = 10_000
num_eval_steps = 5_000

seed = 0
num_seeds = 1

env, env_params = environment.make("gymnax::Pendulum-v1")

cfg = PPOConfig(
    name="PPO",
    num_envs=32,
    num_eval_envs=16,
    num_steps=64,
    gamma=0.999,
    gae_lambda=0.95,
    num_minibatches=8,
    update_epochs=4,
    normalize_advantage=True,
    clip_coef=0.2,
    clip_vloss=True,
    ent_coef=0.01,
    vf_coef=0.5,
)

actor_network = Network(
    feature_extractor=FeatureExtractor(
        observation_extractor=MLP(
            features=(192,), kernel_init=nn.initializers.orthogonal(scale=1.414)
        ),
    ),
    torso=RNN(cell=nn.GRUCell(features=256)),
    head=heads.Gaussian(
        action_dim=env.action_space(env_params).shape[0],
    ),
)
actor_optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adam(learning_rate=3e-4, eps=1e-5),
)

critic_network = Network(
    feature_extractor=FeatureExtractor(
        observation_extractor=MLP(
            features=(192,), kernel_init=nn.initializers.orthogonal(scale=1.414)
        ),
    ),
    torso=RNN(cell=nn.GRUCell(features=256)),
    head=heads.VNetwork(),
)
critic_optimizer = optax.chain(
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
    actor_optimizer=actor_optimizer,
    critic_optimizer=critic_optimizer,
)

logger = Logger(
    [
        DashboardLogger(title="PPO bsuite Example", total_timesteps=total_timesteps),
        # WandbLogger(entity="noahfarr", project="memorax", name="PPO_bsuite"),
    ]
)
logger_state = logger.init(cfg=cfg)

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

    SPS = int(num_train_steps / (end - start))

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
