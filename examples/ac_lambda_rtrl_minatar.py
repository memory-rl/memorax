import time
from dataclasses import asdict

import flax.linen as nn
import jax
import lox
from memorax.algorithms import ACLambda, ACLambdaConfig
from memorax.environments import environment
from memorax.environments.wrappers import RecordEpisodeStatistics
from memorax.loggers import DashboardLogger, MultiLogger
from memorax.networks import RTRL, RNN, Flatten, FeatureExtractor, Network, Stack, heads
from memorax.networks.blocks.ffn import Projection
from memorax.networks.sequence_models import RTUCell

total_timesteps = 10_000_000
num_epochs = 100
num_steps = total_timesteps // num_epochs
seed = 0
num_seeds = 1
env_id = "gymnax::Breakout-MinAtar"

env, env_params = environment.make(env_id)
env = RecordEpisodeStatistics(env)

num_actions = env.action_space(env_params).n

config = ACLambdaConfig(
    num_envs=1,
    trace_lambda=0.8,
    actor_lr=1.0,
    critic_lr=1.0,
    actor_kappa=3.0,
    critic_kappa=2.0,
    entropy_coefficient=0.01,
)

feature_extractor = FeatureExtractor(
    observation_extractor=nn.Sequential(
        (
            nn.Conv(16, (3, 3), strides=(1, 1), padding="VALID"),
            nn.LayerNorm(use_scale=False, use_bias=False),
            nn.leaky_relu,
            Flatten(start_dim=-3),
            nn.Dense(128),
            nn.LayerNorm(use_scale=False, use_bias=False),
            nn.leaky_relu,
        )
    ),
)

actor_network = Network(
    feature_extractor=feature_extractor,
    torso=Stack(
        blocks=(
            RTRL(sequence_model=RNN(cell=RTUCell(features=128, hidden_dim=128))),
            Projection(features=128),
        )
    ),
    head=heads.Categorical(action_dim=num_actions),
)

critic_network = Network(
    feature_extractor=feature_extractor,
    torso=Stack(
        blocks=(
            RTRL(sequence_model=RNN(cell=RTUCell(features=128, hidden_dim=128))),
            Projection(features=128),
        )
    ),
    head=heads.VNetwork(gamma=0.99),
)

agent = ACLambda(config, env, env_params, actor_network, critic_network)

logger = MultiLogger(
    [
        DashboardLogger(
            total_timesteps=total_timesteps,
            summary={"Algorithm": "AC-Lambda", "Environment": env_id, "Torso": "RTRL-RTU", "Total Timesteps": f"{total_timesteps:_}"},
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
