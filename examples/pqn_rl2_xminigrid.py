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
    RL2Wrapper,
    Residual,
    Stack,
    heads,
)
from memorax.networks.blocks.ffn import Projection

total_timesteps = 10_000_000
total_timesteps_decay = 1_000_000
num_epochs = 100
num_steps = total_timesteps // num_epochs
seed = 0
num_seeds = 1
env_id = "xminigrid::MiniGrid-MemoryS8"

env, env_params = environment.make(env_id)
env = RecordEpisodeStatistics(env)

num_actions = env.action_space(env_params).n
num_episodes_per_trial = 10
steps_per_trial = env_params.max_steps_in_episode * num_episodes_per_trial

cfg = PQNConfig(
    num_envs=16,
    num_steps=128,
    td_lambda=0.95,
    num_minibatches=16,
    update_epochs=4,
)

feature_extractor = FeatureExtractor(
    observation_extractor=nn.Sequential(
        (Flatten(start_dim=-3), nn.Dense(128), nn.relu)
    ),
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
            Projection(features=256),
            Residual(
                module=RL2Wrapper(
                    sequence_model=RNN(cell=nn.GRUCell(features=256)),
                    steps_per_trial=steps_per_trial,
                )
            ),
            Residual(
                module=RL2Wrapper(
                    sequence_model=RNN(cell=nn.GRUCell(features=256)),
                    steps_per_trial=steps_per_trial,
                )
            ),
            Projection(features=128),
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
keys = jax.random.split(key, num_seeds)

keys, state = init(keys)

for i in range(num_epochs):
    start = time.perf_counter()
    (keys, state), logs = train(keys, state, num_steps)
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
    logger.log(data, step=state.step[0].item())

logger.finish()
