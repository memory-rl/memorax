import time
from dataclasses import asdict

import flax.linen as nn
import jax
import lox
import optax
from flashbax import make_item_buffer
from gymnax.wrappers import LogWrapper

from memorax.algorithms import DQN, DQNConfig
from memorax.environments import environment
from memorax.loggers import DashboardLogger, Logger
from memorax.networks import FeatureExtractor, Network, heads

total_timesteps = 500_000
num_epochs = 50
num_steps = total_timesteps // num_epochs
seed = 0
num_seeds = 1

env, env_params = environment.make("gxm::Gymnasium/LunarLander-v3")
env = LogWrapper(env)

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

logger = Logger([DashboardLogger(title="DQN GXM", total_timesteps=total_timesteps)])
logger_state = logger.init(cfg=asdict(cfg))

init = jax.vmap(agent.init)
warmup = jax.vmap(agent.warmup, in_axes=(0, 0, None))
train = jax.vmap(lox.spool(agent.train), in_axes=(0, 0, None))

key = jax.random.key(seed)
keys = jax.random.split(key, num_seeds)

keys, state = init(keys)
keys, state = warmup(keys, state, 5_000)

for i in range(num_epochs):
    start = time.perf_counter()
    (keys, state), logs = train(keys, state, num_steps)
    jax.block_until_ready(state)
    end = time.perf_counter()

    SPS = int(num_steps / (end - start))

    print(logs.keys())
    data = {
        "training/SPS": SPS,
        **{f"training/{k}": v for k, v in logs["info"].items()},
    }
    logger_state = logger.log(logger_state, data, step=state.step[0].item())
    logger.emit(logger_state)

logger.finish(logger_state)
