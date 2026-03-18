import time
from dataclasses import asdict

import flax.linen as nn
import jax
import lox
import optax

from memorax.algorithms import PPO, PPOConfig
from memorax.environments import environment
from memorax.environments.wrappers import RecordEpisodeStatistics
from memorax.loggers import DashboardLogger, MultiLogger
from memorax.networks import RNN, FeatureExtractor, Network, Stack, heads
from memorax.networks.blocks.ffn import Projection

total_timesteps = 1_500_000
num_epochs = 10
num_steps = total_timesteps // num_epochs
seed = 0
num_seeds = 1
env_id = "pobax::tmaze_5"

env, env_params = environment.make(env_id)
env = RecordEpisodeStatistics(env)

num_actions = env.action_space(env_params).n

cfg = PPOConfig(
    num_envs=256,
    num_steps=32,
    gae_lambda=0.95,
    num_minibatches=4,
    update_epochs=4,
    normalize_advantage=True,
    clip_coefficient=0.2,
    clip_value_loss=True,
    entropy_coefficient=0.01,
)

feature_extractor = FeatureExtractor(
    observation_extractor=nn.Sequential((nn.Dense(128), nn.relu)),
)

actor_network = Network(
    feature_extractor=feature_extractor,
    torso=Stack(
        blocks=(
            RNN(cell=nn.GRUCell(features=128)),
            Projection(features=128),
        )
    ),
    head=heads.Categorical(action_dim=num_actions),
)

critic_network = Network(
    feature_extractor=feature_extractor,
    torso=Stack(
        blocks=(
            RNN(cell=nn.GRUCell(features=128)),
            Projection(features=128),
        )
    ),
    head=heads.VNetwork(),
)

num_updates = total_timesteps // (cfg.num_steps * cfg.num_envs)

learning_rate = optax.linear_schedule(
    init_value=2.5e-4,
    end_value=0.0,
    transition_steps=num_updates * cfg.update_epochs * cfg.num_minibatches,
)

optimizer = optax.chain(
    optax.clip_by_global_norm(0.5),
    optax.adam(learning_rate, eps=1e-5),
)

agent = PPO(cfg, env, env_params, actor_network, critic_network, optimizer, optimizer)

logger = MultiLogger(
    [
        DashboardLogger(
            total_timesteps=total_timesteps,
            summary={"Algorithm": "PPO", "Environment": env_id, "Torso": "GRU", "Total Timesteps": f"{total_timesteps:_}"},
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
