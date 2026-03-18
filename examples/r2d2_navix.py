import time
from dataclasses import asdict
from functools import partial

import flax.linen as nn
import jax
import lox
import optax

from memorax.algorithms import R2D2, R2D2Config
from memorax.buffers import make_prioritised_episode_buffer
from memorax.environments import environment
from memorax.environments.wrappers import RecordEpisodeStatistics
from memorax.loggers import DashboardLogger, MultiLogger
from memorax.networks import (
    RNN,
    FeatureExtractor,
    Flatten,
    Network,
    Residual,
    Stack,
    heads,
)
from memorax.networks.blocks.ffn import Projection

total_timesteps = 2_000_000
num_epochs = 50
num_steps = total_timesteps // num_epochs
warmup_steps = 10_000
seed = 0
num_seeds = 1

env_id = "navix::Navix-DoorKey-5x5-v0"
env, env_params = environment.make(env_id)
env = RecordEpisodeStatistics(env)

num_actions = env.action_space(env_params).n

buffer_size = 100_000
burn_in_length = 10
sequence_length = 40

cfg = R2D2Config(
    num_envs=16,
    tau=1.0,
    target_update_frequency=2500,
    train_frequency=16,
    burn_in_length=burn_in_length,
    sequence_length=sequence_length,
    n_step=5,
    priority_exponent=0.9,
    importance_sampling_exponent=0.6,
)

feature_extractor = FeatureExtractor(
    observation_extractor=nn.Sequential(
        (Flatten(start_dim=-3), nn.Dense(256), nn.relu, nn.Dense(256), nn.relu)
    ),
    action_extractor=partial(jax.nn.one_hot, num_classes=num_actions),
)

optimizer = optax.chain(
    optax.clip_by_global_norm(0.5),
    optax.adam(1e-4),
)

epsilon = optax.linear_schedule(
    1.0,
    0.01,
    int(total_timesteps * 0.5),
)

beta = optax.linear_schedule(
    cfg.importance_sampling_exponent,
    1.0,
    total_timesteps,
)

q_network = Network(
    feature_extractor=feature_extractor,
    torso=Stack(
        blocks=(
            Projection(features=256),
            Residual(module=RNN(cell=nn.GRUCell(features=256))),
            Projection(features=256),
        )
    ),
    head=heads.DiscreteQNetwork(action_dim=num_actions),
)

buffer = make_prioritised_episode_buffer(
    max_length=buffer_size,
    min_length=256,
    sample_batch_size=32,
    sample_sequence_length=sequence_length,
    add_batch_size=cfg.num_envs,
    add_sequences=True,
    priority_exponent=cfg.priority_exponent,
    device="cpu",
)

agent = R2D2(
    cfg,
    env,
    env_params,
    q_network,
    optimizer,
    buffer,
    epsilon,
    beta,
)

logger = MultiLogger(
    [
        DashboardLogger(
            total_timesteps=total_timesteps,
            summary={"Algorithm": "R2D2", "Environment": env_id, "Torso": "GRU", "Total Timesteps": f"{total_timesteps:_}"},
        )
    ]
)

init = jax.vmap(agent.init)
warmup = jax.vmap(agent.warmup, in_axes=(0, 0, None))
train = jax.vmap(lox.spool(agent.train), in_axes=(0, 0, None))

key = jax.random.key(seed)
key, init_key = jax.random.split(key)
state = init(jax.random.split(init_key, num_seeds))
key, warmup_key = jax.random.split(key)
state = warmup(jax.random.split(warmup_key, num_seeds), state, warmup_steps)

for i in range(num_epochs):
    start = time.perf_counter()
    key, train_key = jax.random.split(key)
    state, logs = train(jax.random.split(train_key, num_seeds), state, num_steps)
    jax.block_until_ready(state)
    end = time.perf_counter()

    SPS = int(num_steps / (end - start))

    info = logs.pop("info")
    episode_returns = info["returned_episode_returns"][info["returned_episode"]]
    episode_lengths = info["returned_episode_lengths"][info["returned_episode"]]

    data = {
        "training/SPS": SPS,
        "training/episode_returns": episode_returns,
        "training/episode_lengths": episode_lengths,
        **logs,
    }
    logger.log(data, step=state.step.mean().item())

logger.finish()
