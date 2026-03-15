import time
from dataclasses import asdict

import flax.linen as nn
import jax
import lox
import optax
from memorax.algorithms import PPO, PPOConfig
from memorax.environments import environment
from memorax.environments.wrappers import RecordEpisodeStatistics
from memorax.loggers import DashboardLogger, Logger
from memorax.networks import RNN, FeatureExtractor, Network, Stack, heads
from memorax.networks.blocks.ffn import Projection

total_timesteps = 15_000_000
num_epochs = 150
num_steps = total_timesteps // num_epochs
seed = 0
num_seeds = 1
env_id = "popjym::CountRecallEasy"

env, env_params = environment.make(env_id)
env = RecordEpisodeStatistics(env)

num_actions = env.action_space(env_params).n

cfg = PPOConfig(
    num_envs=64,
    num_steps=1024,
    gae_lambda=1.0,
    num_minibatches=8,
    update_epochs=30,
    normalize_advantage=True,
    clip_coefficient=0.3,
    clip_value_loss=True,
    entropy_coefficient=0.0,
    target_kl=0.01,
)

feature_extractor = FeatureExtractor(
    observation_extractor=nn.Sequential(
        (nn.Dense(256), nn.LayerNorm(), nn.leaky_relu)
    ),
)

optimizer = optax.chain(
    optax.clip_by_global_norm(0.5),
    optax.adam(5e-5),
)

actor_network = Network(
    feature_extractor=feature_extractor,
    torso=Stack(
        blocks=(
            RNN(cell=nn.GRUCell(features=256)),
            Projection(features=256),
        )
    ),
    head=heads.Categorical(action_dim=num_actions),
)

critic_network = Network(
    feature_extractor=feature_extractor,
    torso=Stack(
        blocks=(
            RNN(cell=nn.GRUCell(features=256)),
            Projection(features=256),
        )
    ),
    head=heads.VNetwork(),
)

agent = PPO(cfg, env, env_params, actor_network, critic_network, optimizer, optimizer)

logger = Logger(
    [
        DashboardLogger(
            title="PPO PopJym",
            name="PPO",
            env_id=env_id,
            total_timesteps=total_timesteps,
        )
    ]
)
logger_state = logger.init(cfg=asdict(cfg))

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
    logger_state = logger.log(logger_state, data, step=state.step[0].item())
    logger.emit(logger_state)

logger.finish(logger_state)
