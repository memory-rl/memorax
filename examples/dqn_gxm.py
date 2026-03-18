import time
from dataclasses import asdict

import flax.linen as nn
import jax
import lox
import optax
from flashbax import make_item_buffer

from memorax.algorithms import DQN, DQNConfig
from memorax.environments import environment
from memorax.environments.wrappers import RecordEpisodeStatistics
from memorax.loggers import DashboardLogger, MultiLogger
from memorax.networks import FeatureExtractor, Network, heads

total_timesteps = 500_000
num_epochs = 50
num_steps = total_timesteps // num_epochs
seed = 0
num_seeds = 1

env_id = "gxm::Gymnasium/LunarLander-v3"
env, env_params = environment.make(env_id)
env = RecordEpisodeStatistics(env)

cfg = DQNConfig(
    num_envs=10,
    tau=1.0,
    target_update_frequency=500,
    train_frequency=10,
)

feature_extractor = FeatureExtractor(
    observation_extractor=nn.Sequential(
        (nn.Dense(120), nn.relu, nn.Dense(84), nn.relu)
    ),
)

q_network = Network(
    feature_extractor=feature_extractor,
    head=heads.DiscreteQNetwork(action_dim=env.action_space(env_params).n),
)

optimizer = optax.adam(3e-4)
buffer = make_item_buffer(
    max_length=10_000,
    min_length=64,
    sample_batch_size=64,
    add_sequences=True,
    add_batches=True,
)
epsilon = optax.linear_schedule(
    1.0,
    0.05,
    int(total_timesteps * 0.5),
)

agent = DQN(cfg, env, env_params, q_network, optimizer, buffer, epsilon)

logger = MultiLogger(
    [
        DashboardLogger(
            total_timesteps=total_timesteps,
            summary={"Algorithm": "DQN", "Environment": env_id, "Torso": "MLP", "Total Timesteps": f"{total_timesteps:_}"},
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
state = warmup(jax.random.split(warmup_key, num_seeds), state, 5_000)

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
