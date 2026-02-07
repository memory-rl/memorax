import time
from dataclasses import asdict

import flax.linen as nn
import jax

# GPU optimization: Use TensorFloat32 for faster matmuls on modern NVIDIA GPUs (L4, A100, etc.)
jax.config.update("jax_default_matmul_precision", "tensorfloat32")
# Disable PGLE (Profile Guided Latency Estimator) which uses CUDA graph capture
# This is required because pure_callback and scatter operations don't support graph capture
jax.config.update("jax_pgle_profiling_runs", 0)
import optax

import memorax.environments.cogames  # registers "cogames"
from memorax.algorithms import IMPALA, IMPALAConfig
from memorax.environments import pufferlib as pufferlib_env
from memorax.environments.cogames import SingleAgentBoxObsWrapper
from memorax.loggers import DashboardLogger, Logger
from memorax.networks import CNN, MLP, FeatureExtractor, Network, heads


total_timesteps = 50_000_000
min_steps_per_env = 10_000  # Minimum steps per env per training call

seed = 0
num_envs = 8  # Each env has ~100 agents, so effective batch = num_envs * agents_per_env

env, env_info = pufferlib_env.make(
    "cogames:cogsguard_arena.basic",
    num_envs=num_envs,
    variants=["credit"],
)
env = SingleAgentBoxObsWrapper(env)
env_params = env.default_params


cfg = IMPALAConfig(
    name="IMPALA",
    num_envs=env.num_envs,  # num_envs * agents_per_env (PufferLib flattens agents into env dim)
    num_eval_envs=0,
    num_steps=64,
    gamma=0.99,
    gae_lambda=0.95,
    num_minibatches=4,
    normalize_advantage=True,
    ent_coef=0.01,
    vf_coef=0.5,
)

num_train_steps = 10_000 * env.num_envs

d_model = 128

observation_extractor = CNN(
    features=(128, 128),
    kernel_sizes=((5, 5), (3, 3)),
    strides=(3, 1),
    normalize=False,  # SingleAgentBoxObsWrapper already normalizes to [0, 1]
)

feature_extractor = FeatureExtractor(
    observation_extractor=observation_extractor,
)
torso = MLP(features=(128,))
actor_network = nn.remat(Network)(
    feature_extractor=feature_extractor,
    torso=torso,
    head=heads.Categorical(
        action_dim=env.num_actions,
    ),
)

critic_network = nn.remat(Network)(
    feature_extractor=feature_extractor,
    torso=torso,
    head=heads.VNetwork(),
)

optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adam(learning_rate=3e-4, eps=1e-5),
)


key = jax.random.key(seed)

agent = IMPALA(
    cfg=cfg,
    env=env,
    env_params=env_params,
    actor_network=actor_network,
    critic_network=critic_network,
    actor_optimizer=optimizer,
    critic_optimizer=optimizer,
)

logger = Logger(
    [DashboardLogger(title="IMPALA PufferLib CoGames", total_timesteps=total_timesteps)]
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
    info = jax.tree.map(lambda x: x.mean(), transitions.info)
    info = jax.tree.map(lambda x: x[None], info)
    info = {f"training/{k}": v for k, v in info.items()}
    data = {"training/SPS": SPS, **training_statistics, **info}
    logger_state = logger.log(logger_state, data, step=state.step.item())
    logger.emit(logger_state)
