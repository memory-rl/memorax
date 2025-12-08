import time

import jax
import flax.linen as nn
import optax
from memorax.algorithms.drqn import DRQN, DRQNConfig
from memorax.buffers import make_episode_buffer
from memorax.environments import environment
from memorax.loggers import Logger, DashboardLogger
from memorax.networks import (
    MLP,
    SequenceNetwork,
    RNN,
    heads,
    FeatureExtractor,
)

total_timesteps = 500_000
num_train_steps = 50_000
num_eval_steps = 5_000

env, env_params = environment.make("gymnax::MemoryChain-bsuite")

cfg = DRQNConfig(
    name="drqn",
    learning_rate=3e-4,
    num_envs=10,
    num_eval_envs=10,
    buffer_size=10_000,
    gamma=0.99,
    tau=1.0,
    target_network_frequency=500,
    batch_size=16,
    start_e=1.0,
    end_e=0.05,
    exploration_fraction=0.5,
    learning_starts=10_000,
    train_frequency=10,
    double=False,
    sequence_length=6,
    burn_in_length=0,
    mask=False,
)

q_network = SequenceNetwork(
    feature_extractor=FeatureExtractor(observation_extractor=MLP(features=(128,))),
    torso=RNN(cell=nn.GRUCell(features=128), unroll=16),
    head=heads.DiscreteQNetwork(
        action_dim=env.action_space(env_params).n,
    ),
)

optimizer = optax.chain(
    optax.clip_by_global_norm(5.0),
    optax.adam(learning_rate=cfg.learning_rate, eps=1e-5),
)

buffer = make_episode_buffer(
    max_length=cfg.buffer_size,
    min_length=cfg.batch_size,
    sample_batch_size=cfg.batch_size,
    add_sequences=True,
    add_batch_size=cfg.num_envs,
    sample_sequence_length=env_params.max_steps_in_episode,
    min_length_time_axis=env_params.max_steps_in_episode,
)

epsilon_schedule = optax.linear_schedule(
    cfg.start_e,
    cfg.end_e,
    int(cfg.exploration_fraction * total_timesteps),
    cfg.learning_starts,
)

key = jax.random.key(0)

agent = DRQN(
    cfg=cfg,
    env=env,
    env_params=env_params,
    q_network=q_network,
    optimizer=optimizer,
    buffer=buffer,
    epsilon_schedule=epsilon_schedule,
)

logger = Logger(
    [DashboardLogger(title="DRQN Example", total_timesteps=total_timesteps)]
)
logger_state = logger.init(cfg)

key, state = agent.init(key)
key, transitions = agent.evaluate(key, state, num_steps=num_eval_steps)
evaluation_statistics = Logger.get_episode_statistics(transitions, "evaluation")
logger_state = logger.log(logger_state, evaluation_statistics, step=state.step.item())
logger.emit(logger_state)

for i in range(0, total_timesteps, num_train_steps):

    start = time.perf_counter()
    key, state, transitions = agent.train(key, state, num_steps=num_train_steps)
    end = time.perf_counter()

    SPS = int(num_train_steps / (end - start))

    training_statistics = Logger.get_episode_statistics(transitions, "training")
    data = {"SPS": SPS, **training_statistics, **transitions.losses}
    logger_state = logger.log(logger_state, data, step=state.step.item())

    key, transitions = agent.evaluate(key, state, num_steps=num_eval_steps)
    evaluation_statistics = Logger.get_episode_statistics(transitions, "evaluation")
    logger_state = logger.log(
        logger_state, evaluation_statistics, step=state.step.item()
    )
    logger.emit(logger_state)
