import time
from functools import partial

import flax.linen as nn
import gymnasium
import jax
import lox
import optax
import pufferlib.emulation
import pufferlib.vector

from memorax.algorithms import PQN, PQNConfig
from memorax.environments.pufferlib import make
from memorax.environments.wrappers import RecordEpisodeStatistics
from memorax.loggers import DashboardLogger, MultiLogger
from memorax.networks import RNN, FeatureExtractor, Network, Residual, Stack, heads
from memorax.networks.blocks.ffn import Projection

total_timesteps = 500_000
total_timesteps_decay = 250_000
num_epochs = 50
num_steps = total_timesteps // num_epochs
seed = 0
num_envs = 64
env_id = "CartPole-v1"

env, env_params = make(
    env_id,
    env_creator=pufferlib.emulation.GymnasiumPufferEnv,
    env_kwargs={"env_creator": lambda: gymnasium.make(env_id)},
    num_envs=num_envs,
    backend=pufferlib.vector.Multiprocessing,
    num_workers=16,
)
env = RecordEpisodeStatistics(env)

num_actions = env.action_space(env_params).n

cfg = PQNConfig(
    num_envs=num_envs,
    num_steps=128,
    td_lambda=0.95,
    num_minibatches=16,
    update_epochs=4,
)

feature_extractor = FeatureExtractor(
    observation_extractor=nn.Sequential(
        (nn.Dense(120), nn.relu, nn.Dense(84), nn.relu)
    ),
    action_extractor=partial(jax.nn.one_hot, num_classes=num_actions),
)

num_updates_decay = total_timesteps_decay // (cfg.num_steps * cfg.num_envs)

optimizer = optax.chain(
    optax.clip_by_global_norm(0.5),
    optax.radam(learning_rate=3e-4),
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
            Projection(features=128),
            Residual(module=RNN(cell=nn.GRUCell(features=128))),
            Projection(features=64),
        )
    ),
    head=heads.DiscreteQNetwork(action_dim=num_actions),
)

agent = PQN(cfg, env, env_params, q_network, optimizer, epsilon)

logger = MultiLogger(
    [
        DashboardLogger(
            total_timesteps=total_timesteps,
            summary={
                "Algorithm": "PQN",
                "Environment": env_id,
                "Torso": "GRU",
                "Total Timesteps": f"{total_timesteps:_}",
            },
        )
    ]
)

init = jax.jit(agent.init)
train = lox.spool(jax.jit(agent.train, static_argnames=["num_steps"]))

key = jax.random.key(seed)

key, init_key = jax.random.split(key)
state = init(init_key)

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
