import time
from dataclasses import asdict

import flax.linen as nn
import jax
import optax

from memorax.algorithms import PQN, PQNConfig
from memorax.environments import environment
from memorax.loggers import DashboardLogger, Logger
from memorax.networks import FeatureExtractor, Network, heads

total_timesteps = 1_000_000
num_train_steps = 10_000
num_eval_steps = 10_000

seed = 0
num_seeds = 1

env, env_params = environment.make("pobax::tmaze_5")

cfg = PQNConfig(
    num_envs=32,
    num_steps=64,
    td_lambda=0.95,
    num_minibatches=4,
    update_epochs=4,
)

q_network = Network(
    feature_extractor=FeatureExtractor(
        observation_extractor=nn.Sequential(
            (
                nn.Dense(64, kernel_init=nn.initializers.orthogonal(scale=1.414)),
                nn.relu,
                nn.Dense(64, kernel_init=nn.initializers.orthogonal(scale=1.414)),
                nn.relu,
            )
        ),
    ),
    head=heads.DiscreteQNetwork(
        action_dim=env.action_space(env_params).n,
    ),
)

optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adam(learning_rate=3e-4, eps=1e-5),
)

epsilon_schedule = optax.linear_schedule(
    1.0,
    0.05,
    int(0.1 * total_timesteps),
)

key = jax.random.key(seed)
keys = jax.random.split(key, num_seeds)

agent = PQN(
    cfg=cfg,
    env=env,
    env_params=env_params,
    q_network=q_network,
    optimizer=optimizer,
    epsilon_schedule=epsilon_schedule,
)

logger = Logger(
    [
        DashboardLogger(
            title="PQN PoBax TMaze-5",
            total_timesteps=total_timesteps,
            env_id="pobax::tmaze_5",
        ),
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
