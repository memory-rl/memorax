import time
from dataclasses import asdict

import flax.linen as nn
import jax
import lox
import optax

from memorax.algorithms.gradient_ppo import GradientPPO, GradientPPOConfig
from memorax.environments import environment
from memorax.environments.wrappers import RecordEpisodeStatistics
from memorax.loggers import DashboardLogger, MultiLogger
from memorax.networks import (
    FeatureExtractor,
    MambaCell,
    Memoroid,
    Network,
    Stack,
    heads,
)
from memorax.networks.blocks.ffn import Projection

total_timesteps = 1_000_000_000
num_epochs = 1000
num_steps = total_timesteps // num_epochs
seed = 0
num_seeds = 1
env_id = "craftax::Craftax-Symbolic-v1"

env, env_params = environment.make(env_id, auto_reset=True)
env = RecordEpisodeStatistics(env)

num_actions = env.action_space(env_params).n

config = GradientPPOConfig(
    num_envs=1024,
    num_steps=64,
    gae_lambda=0.8,
    num_minibatches=8,
    update_epochs=4,
    normalize_advantage=True,
    clip_coefficient=0.2,
    clip_value_loss=True,
    entropy_coefficient=0.01,
    regularization_coefficient=1e-4,
    truncation_length=64,
)

feature_extractor = FeatureExtractor(
    observation_extractor=nn.Sequential(
        (nn.Dense(512), nn.tanh, nn.Dense(512), nn.tanh)
    ),
)

num_updates = total_timesteps // (config.num_steps * config.num_envs)

learning_rate = optax.linear_schedule(
    init_value=2e-4,
    end_value=0.0,
    transition_steps=num_updates * config.update_epochs * config.num_minibatches,
)

optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adam(learning_rate, eps=1e-5),
)

actor_network = Network(
    feature_extractor=feature_extractor,
    torso=Stack(
        blocks=(
            Memoroid(cell=MambaCell(features=512)),
            Projection(features=512),
        )
    ),
    head=heads.Categorical(action_dim=num_actions),
)

critic_network = Network(
    feature_extractor=feature_extractor,
    torso=Stack(
        blocks=(
            Memoroid(cell=MambaCell(features=512)),
            Projection(features=512),
        )
    ),
    head=heads.VNetwork(gamma=0.99),
)

h_network = Network(
    feature_extractor=feature_extractor,
    torso=Stack(
        blocks=(
            Memoroid(cell=MambaCell(features=512)),
            Projection(features=512),
        )
    ),
    head=heads.VNetwork(gamma=0.99),
)

agent = GradientPPO(
    config,
    env,
    env_params,
    actor_network,
    critic_network,
    h_network,
    optimizer,
    optimizer,
    optimizer,
)

logger = MultiLogger(
    [
        DashboardLogger(
            total_timesteps=total_timesteps,
            summary={"Algorithm": "GradientPPO", "Environment": env_id, "Torso": "Mamba", "Total Timesteps": f"{total_timesteps:_}"},
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
