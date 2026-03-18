import time
from dataclasses import asdict
from functools import partial

import flax.linen as nn
import jax
import jax.numpy as jnp
import lox
import optax
from memorax.algorithms import PQN, PQNConfig
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


class CNN(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = x.astype(jnp.float32) / 255.0
        x = nn.leaky_relu(nn.Conv(64, (5, 5), strides=(2, 2), padding="VALID")(x))
        x = nn.max_pool(x, (2, 2), strides=(2, 2))
        x = nn.leaky_relu(nn.Conv(128, (3, 3), strides=(2, 2), padding="VALID")(x))
        x = nn.max_pool(x, (2, 2), strides=(2, 2))
        x = nn.leaky_relu(nn.Conv(256, (3, 3), strides=(2, 2), padding="VALID")(x))
        x = nn.max_pool(x, (3, 3), strides=(1, 1))
        x = nn.leaky_relu(nn.Conv(512, (1, 1), strides=(1, 1), padding="VALID")(x))
        x = Flatten(start_dim=-3)(x)
        return x


total_timesteps = 10_000_000
total_timesteps_decay = 1_000_000
num_epochs = 100
num_steps = total_timesteps // num_epochs
seed = 0
num_seeds = 1
env_id = "popgym_arcade::CountRecallEasy"

env, env_params = environment.make(env_id)
env = RecordEpisodeStatistics(env)

num_actions = env.action_space(env_params).n

cfg = PQNConfig(
    num_envs=16,
    num_steps=128,
    td_lambda=0.95,
    num_minibatches=16,
    update_epochs=4,
)

feature_extractor = FeatureExtractor(
    observation_extractor=CNN(),
    action_extractor=partial(jax.nn.one_hot, num_classes=num_actions),
)

num_updates_decay = total_timesteps_decay // (cfg.num_steps * cfg.num_envs)

learning_rate = optax.linear_schedule(
    init_value=5e-5,
    end_value=1e-20,
    transition_steps=num_updates_decay * cfg.num_minibatches * cfg.update_epochs,
)

optimizer = optax.chain(
    optax.clip_by_global_norm(0.5),
    optax.radam(learning_rate),
)

epsilon = optax.linear_schedule(
    1.0,
    0.05,
    int(0.25 * total_timesteps_decay),
)

q_network = Network(
    feature_extractor=feature_extractor,
    torso=Stack(
        blocks=(
            Projection(features=512),
            Residual(module=RNN(cell=nn.GRUCell(features=512))),
            Residual(module=RNN(cell=nn.GRUCell(features=512))),
            Projection(features=256),
        )
    ),
    head=heads.DiscreteQNetwork(action_dim=num_actions),
)

agent = PQN(cfg, env, env_params, q_network, optimizer, epsilon)

logger = MultiLogger(
    [
        DashboardLogger(
            total_timesteps=total_timesteps,
            summary={"Algorithm": "PQN", "Environment": env_id, "Torso": "GRU", "Total Timesteps": f"{total_timesteps:_}"},
        )
    ]
)

init = jax.vmap(agent.init)
train = jax.vmap(lox.spool(agent.train), in_axes=(0, 0, None))

key = jax.random.key(seed)
key, init_key = jax.random.split(key)
state = init(jax.random.split(init_key, num_seeds))

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
