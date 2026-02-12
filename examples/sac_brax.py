import time
from dataclasses import asdict

import flax.linen as nn
import jax
import optax
from flashbax import make_item_buffer

from memorax.algorithms import SAC, SACConfig
from memorax.environments import environment
from memorax.loggers import DashboardLogger, Logger
from memorax.networks import FeatureExtractor, Network, heads

total_timesteps = 6_553_600
num_train_steps = 65_536
num_eval_steps = 5_000

seed = 0
num_seeds = 1

env, env_params = environment.make("brax::hopper", mode="V")


cfg = SACConfig(
    actor_lr=6e-4,
    critic_lr=6e-4,
    alpha_lr=6e-4,
    num_envs=128,
    num_eval_envs=128,
    buffer_size=1_048_576,
    gamma=0.997,
    tau=0.005,
    target_update_frequency=1,
    batch_size=512,
    initial_alpha=1.0,
    target_entropy_scale=1.0,
    max_grad_norm=0.5,
    train_frequency=128,
    gradient_steps=64,
)

actor_network = Network(
    feature_extractor=FeatureExtractor(
        observation_extractor=nn.Sequential((nn.Dense(256), nn.relu, nn.Dense(256), nn.relu)),
    ),
    head=heads.SquashedGaussian(
        action_dim=env.action_space(env_params).shape[0],
    ),
)
actor_optimizer = optax.adam(learning_rate=cfg.actor_lr)

critic_network = nn.vmap(
    Network,
    variable_axes={"params": 0},
    split_rngs={"params": True},
    in_axes=None,
    out_axes=0,
    axis_size=2,
)(
    feature_extractor=FeatureExtractor(
        observation_extractor=nn.Sequential((nn.Dense(256), nn.relu, nn.Dense(256), nn.relu)),
    ),
    head=heads.ContinuousQNetwork(),
)

critic_optimizer = optax.adam(learning_rate=cfg.critic_lr)

alpha_network = heads.Alpha(
    initial_alpha=cfg.initial_alpha,
)

alpha_optimizer = optax.adam(learning_rate=cfg.alpha_lr)

buffer = make_item_buffer(
    max_length=cfg.buffer_size,
    min_length=8192,
    sample_batch_size=cfg.batch_size,
    add_sequences=True,
    add_batches=True,
)

key = jax.random.key(seed)
keys = jax.random.split(key, num_seeds)

agent = SAC(
    cfg=cfg,
    env=env,
    env_params=env_params,
    actor_network=actor_network,
    critic_network=critic_network,
    alpha_network=alpha_network,
    actor_optimizer=actor_optimizer,
    critic_optimizer=critic_optimizer,
    alpha_optimizer=alpha_optimizer,
    buffer=buffer,
)

logger = Logger(
    [
        DashboardLogger(title="SAC Brax Hopper", total_timesteps=total_timesteps, name="SAC", env_id="brax::hopper"),
    ]
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
    infos = jax.vmap(lambda t: t.infos)(transitions)
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
