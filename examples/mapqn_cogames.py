import time
from dataclasses import asdict

import flax.linen as nn
import jax

# GPU optimization
jax.config.update("jax_default_matmul_precision", "tensorfloat32")
jax.config.update("jax_pgle_profiling_runs", 0)
import optax
import pufferlib

import memorax.environments.cogames  # registers "cogames"
from memorax.environments.cogames import BoxObsWrapper
from memorax.algorithms import MAPQN, MAPQNConfig
from memorax.environments import pufferlib as pufferlib_env
from memorax.loggers import DashboardLogger, Logger
from memorax.networks import CNN, MLP, FeatureExtractor, Network, heads

total_timesteps = 10_000_000
num_eval_steps = 1_000

seed = 0
num_envs = 128
num_steps = 64
num_workers = 16
num_train_steps = num_envs * num_steps

env, env_info = pufferlib_env.make(
    "cogames:cogsguard_arena.basic",
    num_envs=num_envs,
    variants=["credit", "milestones"],
    difficulty="easy",
    multi_agent=True,
    backend=pufferlib.vector.Multiprocessing,
    num_workers=num_workers,
)
# Wrap to convert token obs to spatial grid (11, 11, 24)
env = BoxObsWrapper(env)
cfg = MAPQNConfig(
    name="MAPQN",
    num_envs=num_envs,
    num_eval_envs=num_envs,
    num_steps=num_steps,
    gamma=0.99,
    td_lambda=0.95,
    num_minibatches=8,   # larger batches = better GPU utilization
    update_epochs=2,     # fewer epochs = faster
)

d_model = 128

# CNN for spatial grid observations (11, 11, 24)
observation_extractor = CNN(
    features=(32, d_model),
    kernel_sizes=((3, 3), (3, 3)),
    strides=(2, 2),  # stride 2 = smaller output, faster
    padding="SAME",
)

feature_extractor = FeatureExtractor(
    observation_extractor=observation_extractor,
)
torso = MLP(features=(d_model,))

action_space = env.action_spaces[env.agents[0]]

VmappedNetwork = nn.vmap(
    Network,
    variable_axes={"params": None},
    split_rngs={"params": False, "memory": True, "dropout": True},
    in_axes=(0, 0, 0, 0, 0, 0),
    out_axes=(0, 0),
)

q_network = VmappedNetwork(
    feature_extractor=feature_extractor,
    torso=torso,
    head=heads.DiscreteQNetwork(
        action_dim=action_space.n,
    ),
)

optimizer = optax.contrib.muon(
    learning_rate=3e-4,
)

epsilon_schedule = optax.linear_schedule(
    1.0,
    0.05,
    int(0.5 * total_timesteps),
)

key = jax.random.key(seed)

agent = MAPQN(
    cfg=cfg,
    env=env,
    q_network=q_network,
    optimizer=optimizer,
    epsilon_schedule=epsilon_schedule,
)

logger = Logger(
    [
        DashboardLogger(
            title="MAPQN CoGames CogsGuard",
            name=cfg.name,
            env_id="cogames:cogsguard_arena.basic",
            total_timesteps=total_timesteps,
        )
    ]
)
logger_state = logger.init(cfg=asdict(cfg))

key, state = agent.init(key)

for i in range(0, total_timesteps, num_train_steps):
    start = time.perf_counter()
    key, state, transitions = agent.train(key, state, num_train_steps)
    jax.block_until_ready(state)
    end = time.perf_counter()

    SPS = int(num_train_steps / (end - start))

    training_statistics = Logger.get_episode_statistics(transitions, "training")
    training_statistics = jax.tree.map(lambda x: x[None], training_statistics)
    losses = jax.tree.map(lambda x: x.mean(), transitions.losses)
    losses = jax.tree.map(lambda x: x[None], losses)
    data = jax.device_get({"training/SPS": SPS, **training_statistics, **losses})
    step = state.step.item()
    del transitions
    logger_state = logger.log(logger_state, data, step=step)
    logger.emit(logger_state)
