import time
from dataclasses import asdict
from typing import Any

import flax.linen as nn
import jax
import jax.numpy as jnp
import lox
import optax
from flax import struct

from memorax.algorithms import MAPPO, MAPPOConfig
from memorax.environments import MultiAgentRecordEpisodeStatistics, environment
from memorax.loggers import DashboardLogger, MultiLogger
from memorax.networks import RNN, FeatureExtractor, Network, Stack, heads
from memorax.networks.blocks.ffn import Projection

total_timesteps = 10_000_000
num_epochs = 100
num_steps = total_timesteps // num_epochs
seed = 0
num_seeds = 1
env_id = "jaxmarl::MPE_simple_spread_v3"


env, env_params = environment.make(env_id)
env = MultiAgentRecordEpisodeStatistics(env)

num_actions = env.action_spaces[env.agents[0]].n

cfg = MAPPOConfig(
    num_envs=128,
    num_steps=128,
    gae_lambda=0.95,
    num_minibatches=4,
    update_epochs=10,
    normalize_advantage=True,
    clip_coefficient=0.2,
    clip_value_loss=True,
    entropy_coefficient=0.01,
)

feature_extractor = FeatureExtractor(
    observation_extractor=nn.Sequential((nn.Dense(128), nn.LayerNorm(), nn.leaky_relu)),
)

optimizer = optax.chain(
    optax.clip_by_global_norm(0.5),
    optax.adam(3e-4),
)

actor_network = nn.vmap(
    Network,
    variable_axes={"params": None, "intermediates": 0},
    split_rngs={"params": False, "torso": True, "dropout": True},
    in_axes=(0, 0, 0, 0, 0, 0),
    out_axes=0,
)(
    feature_extractor=feature_extractor,
    torso=Stack(
        blocks=(
            RNN(cell=nn.GRUCell(features=128)),
            Projection(features=128),
        )
    ),
    head=heads.Categorical(action_dim=num_actions),
)

critic_network = nn.vmap(
    Network,
    variable_axes={"params": None, "intermediates": 0},
    split_rngs={"params": False, "torso": True, "dropout": True},
    in_axes=(None, 0, 0, 0, 0, 0),
    out_axes=0,
)(
    feature_extractor=FeatureExtractor(
        observation_extractor=nn.Sequential(
            (
                lambda x: jnp.moveaxis(x, 0, -1).reshape(*x.shape[1:-1], -1),
                nn.Dense(128),
                nn.LayerNorm(),
                nn.leaky_relu,
            )
        ),
    ),
    torso=Stack(
        blocks=(
            RNN(cell=nn.GRUCell(features=128)),
            Projection(features=128),
        )
    ),
    head=heads.VNetwork(),
)

agent = MAPPO(cfg, env, env_params, actor_network, critic_network, optimizer, optimizer)

logger = MultiLogger(
    [
        DashboardLogger(
            total_timesteps=total_timesteps,
            summary={"Algorithm": "MAPPO", "Environment": env_id, "Torso": "GRU", "Total Timesteps": f"{total_timesteps:_}"},
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
