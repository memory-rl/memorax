import time
from functools import partial

import flax.linen as nn
import jax
import lox
import optax
from flashbax import make_item_buffer

from memorax.algorithms import DQN, DQNConfig
from memorax.environments.atari import make
from memorax.environments.wrappers import RecordEpisodeStatistics
from memorax.loggers import DashboardLogger, MultiLogger
from memorax.networks import RNN, FeatureExtractor, Flatten, Network, Residual, Stack, heads
from memorax.networks.blocks.ffn import Projection

total_timesteps = 10_000_000
num_epochs = 100
num_steps = total_timesteps // num_epochs
seed = 0
num_envs = 16
env_id = "breakout"

env, env_params = make(env_id, num_envs=num_envs)
env = RecordEpisodeStatistics(env)

num_actions = env.action_space(env_params).n

cfg = DQNConfig(
    num_envs=num_envs,
    tau=1.0,
    target_update_frequency=1_000,
    train_frequency=4,
)

feature_extractor = FeatureExtractor(
    observation_extractor=nn.Sequential(
        (
            nn.Conv(32, (8, 8), strides=(4, 4)),
            nn.relu,
            nn.Conv(64, (4, 4), strides=(2, 2)),
            nn.relu,
            nn.Conv(64, (3, 3), strides=(1, 1)),
            nn.relu,
            Flatten(start_dim=-3),
            nn.Dense(512),
            nn.relu,
        )
    ),
    action_extractor=partial(jax.nn.one_hot, num_classes=num_actions),
)

optimizer = optax.adam(1e-4)
buffer = make_item_buffer(
    max_length=100_000,
    min_length=1_000,
    sample_batch_size=32,
    add_sequences=True,
    add_batches=True,
)
epsilon = optax.linear_schedule(
    1.0,
    0.01,
    int(total_timesteps * 0.1),
)

q_network = Network(
    feature_extractor=feature_extractor,
    torso=Stack(
        blocks=(
            Projection(features=512),
            Residual(module=RNN(cell=nn.GRUCell(features=512))),
            Projection(features=256),
        )
    ),
    head=heads.DiscreteQNetwork(action_dim=num_actions),
)

agent = DQN(cfg, env, env_params, q_network, optimizer, buffer, epsilon)

logger = MultiLogger(
    [
        DashboardLogger(
            total_timesteps=total_timesteps,
            summary={
                "Algorithm": "DQN",
                "Environment": env_id,
                "Torso": "GRU",
                "Total Timesteps": f"{total_timesteps:_}",
            },
        )
    ]
)

init = agent.init
warmup = agent.warmup
train = lox.spool(agent.train)

key = jax.random.key(seed)

key, init_key = jax.random.split(key)
state = init(init_key)
key, warmup_key = jax.random.split(key)
state = warmup(warmup_key, state, 10_000)

for i in range(num_epochs):
    start = time.perf_counter()
    key, train_key = jax.random.split(key)
    state, logs = train(train_key, state, num_steps)
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
    logger.log(data, step=state.step.item())

logger.finish()
