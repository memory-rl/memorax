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
from memorax.algorithms import MAPPO, MAPPOConfig
from memorax.environments import pufferlib as pufferlib_env
from memorax.environments.cogames import BoxObsWrapper
from memorax.loggers import DashboardLogger, Logger
from memorax.networks import CNN, RNN, FeatureExtractor, Network, heads

total_timesteps = 10_000_000
num_train_steps = 10_000
num_eval_steps = 1_000

seed = 0
num_envs = 128
num_workers = 16

env, env_info = pufferlib_env.make(
    "cogames:cogsguard_arena.basic",
    num_envs=num_envs,
    variants=["credit"],
    multi_agent=True,
    backend=pufferlib.vector.Multiprocessing,
    num_workers=num_workers,
)
env = BoxObsWrapper(env)

cfg = MAPPOConfig(
    name="MAPPO",
    num_envs=num_envs,
    num_eval_envs=num_envs,
    num_steps=64,
    gamma=0.99,
    gae_lambda=0.95,
    num_minibatches=16,
    update_epochs=4,
    normalize_advantage=True,
    clip_coef=0.2,
    clip_vloss=True,
    ent_coef=0.01,
    vf_coef=0.5,
    centralized_critic=True,
)

d_model = 128

observation_extractor = CNN(
    features=(128, 128),
    kernel_sizes=((5, 5), (3, 3)),
    strides=(3, 1),
    normalize=False,  # BoxObsWrapper already normalizes to [0, 1]
)

feature_extractor = FeatureExtractor(
    observation_extractor=observation_extractor,
)
torso = RNN(cell=nn.GRUCell(features=d_model))

action_space = env.action_spaces[env.agents[0]]

VmappedNetwork = nn.vmap(
    Network,
    variable_axes={"params": None},
    split_rngs={"params": False, "memory": True, "dropout": True},
    in_axes=(0, 0, 0, 0, 0, 0),
    out_axes=(0, 0),
)

actor_network = nn.remat(VmappedNetwork)(
    feature_extractor=feature_extractor,
    torso=torso,
    head=heads.Categorical(
        action_dim=action_space.n,
        kernel_init=nn.initializers.normal(stddev=0.01),
    ),
)

critic_network = nn.remat(Network)(
    feature_extractor=feature_extractor,
    torso=torso,
    head=heads.CentralizedVNetwork(
        num_agents=env.num_agents,
        kernel_init=nn.initializers.xavier_normal(),
    ),
)

optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adam(learning_rate=3e-4, eps=1e-5),
)

key = jax.random.key(seed)

agent = MAPPO(
    cfg=cfg,
    env=env,
    actor_network=actor_network,
    critic_network=critic_network,
    actor_optimizer=optimizer,
    critic_optimizer=optimizer,
)

logger = Logger(
    [DashboardLogger(
        title="MAPPO CoGames CogsGuard",
        name=cfg.name,
        env_id="cogames:cogsguard_arena.basic",
        total_timesteps=total_timesteps,
    )]
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
    data = {"training/SPS": SPS, **training_statistics, **losses}
    logger_state = logger.log(logger_state, data, step=state.step.item())
    logger.emit(logger_state)
