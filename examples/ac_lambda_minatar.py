import time
from dataclasses import asdict

import flax.linen as nn
import jax

from memorax.algorithms import ACLambda, ACLambdaConfig
from memorax.environments import environment
from memorax.loggers import DashboardLogger, Logger
from memorax.networks import FeatureExtractor, Network, heads, initializers
from memorax.environments.wrappers import NormalizeObservationWrapper, NormalizeRewardWrapper

seed = 0
num_seeds = 5

env, env_params = environment.make("gymnax::Breakout-MinAtar")
env = NormalizeObservationWrapper(env)
env = NormalizeRewardWrapper(env, gamma=0.99)

cfg = ACLambdaConfig(
    num_envs=1,
    trace_lambda=0.8,
    actor_lr=1.0,
    critic_lr=1.0,
    actor_kappa=3.0,
    critic_kappa=2.0,
    entropy_coefficient=0.01,
)

feature_extractor = FeatureExtractor(
    observation_extractor=nn.Sequential(
        (
            nn.Conv(features=16, kernel_size=(3, 3), strides=(1, 1), padding='VALID', kernel_init=initializers.sparse_init()),
            nn.LayerNorm(use_bias=False, use_scale=False),
            nn.leaky_relu,
            lambda x: x.reshape((x.shape[0], x.shape[1], -1)),
            nn.Dense(1024, kernel_init=initializers.sparse_init()),
            nn.LayerNorm(use_bias=False, use_scale=False),
            nn.leaky_relu,
            nn.Dense(128, kernel_init=initializers.sparse_init()),
            nn.LayerNorm(use_bias=False, use_scale=False),
            nn.leaky_relu,
        )
    ),
)

actor_network = Network(
    feature_extractor=feature_extractor,
    head=heads.Categorical(
        action_dim=env.action_space(env_params).n,
        kernel_init=initializers.sparse_init(),
    ),
)
critic_network = Network(
    feature_extractor=feature_extractor,
    head=heads.VNetwork(
        kernel_init=initializers.sparse_init(),
    ),
)

key = jax.random.key(seed)
keys = jax.random.split(key, num_seeds)

agent = ACLambda(
    cfg=cfg,
    env=env,
    env_params=env_params,
    actor_network=actor_network,
    critic_network=critic_network,
)

logger = Logger(
    [DashboardLogger(title="AC(λ) MinAtar Breakout", total_timesteps=total_timesteps)]
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

    SPS = int(num_train_steps / (end - start))

    training_statistics = jax.vmap(Logger.get_episode_statistics, in_axes=(0, None))(
        transitions, "training"
    )
    losses = jax.vmap(
        lambda transition: jax.tree.map(lambda x: x.mean(), transition.losses)
    )(transitions)
    infos = jax.vmap(lambda t: t.info)(transitions)
    data = {"training/SPS": SPS, **training_statistics, **losses, **infos}
    logger_state = logger.log(logger_state, data, step=state.step[0].item())

    keys, transitions = evaluate(keys, state, num_eval_steps)
    evaluation_statistics = jax.vmap(Logger.get_episode_statistics, in_axes=(0, None))(
        transitions, "evaluation"
    )
    logger_state = logger.log(
        logger_state, evaluation_statistics, step=state.step[0].item()
    )
    logger.emit(logger_state)

logger.finish(logger_state)
