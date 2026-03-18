import time
from dataclasses import asdict

import flax.linen as nn
import jax
import lox
import optax
from flashbax import make_item_buffer

from memorax.algorithms import SAC, SACConfig
from memorax.environments import environment
from memorax.environments.wrappers import NormalizeObservationWrapper, RecordEpisodeStatistics
from memorax.loggers import DashboardLogger, MultiLogger
from memorax.networks import FeatureExtractor, Network, heads

total_timesteps = 1_000_000
num_epochs = 10
num_steps = total_timesteps // num_epochs
warmup_steps = 50_000
seed = 0
num_seeds = 1

env_id = "brax::halfcheetah"
env, env_params = environment.make(env_id)
env = NormalizeObservationWrapper(env)
env = RecordEpisodeStatistics(env)

action_dim = env.action_space(env_params).shape[0]

cfg = SACConfig(
    num_envs=4,
    tau=0.005,
    train_frequency=4,
    target_update_frequency=1,
    target_entropy_scale=1.0,
    gradient_steps=1,
)

feature_extractor = FeatureExtractor(
    observation_extractor=nn.Sequential(
        (nn.Dense(256), nn.relu, nn.Dense(256), nn.relu)
    ),
)

optimizer = optax.adam(3e-4)

actor_network = Network(
    feature_extractor=feature_extractor,
    head=heads.SquashedGaussian(action_dim=action_dim),
)

critic_network = Network(
    feature_extractor=feature_extractor,
    head=heads.TwinContinuousQNetwork(),
)

alpha_network = heads.Alpha(initial_alpha=1.0)

buffer = make_item_buffer(
    max_length=100_000,
    min_length=256,
    sample_batch_size=256,
    add_sequences=True,
    add_batches=True,
)

agent = SAC(
    cfg,
    env,
    env_params,
    actor_network,
    critic_network,
    alpha_network,
    optimizer,
    optimizer,
    optimizer,
    buffer,
)

logger = MultiLogger(
    [
        DashboardLogger(
            total_timesteps=total_timesteps,
            summary={"Algorithm": "SAC", "Environment": env_id, "Torso": "MLP", "Total Timesteps": f"{total_timesteps:_}"},
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
