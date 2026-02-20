import time
from dataclasses import asdict
from functools import partial

import flax.linen as nn
import jax
import optax
import pufferlib
from cogames.cogs_vs_clips.missions import CogsGuardBasicMission
from cogames.cogs_vs_clips.reward_variants import apply_reward_variants
from mettagrid.envs.mettagrid_puffer_env import MettaGridPufferEnv
from mettagrid.simulator import Simulator

from memorax.algorithms import PPO, PPOConfig
from memorax.loggers import DashboardLogger, Logger
from memorax.networks import FeatureExtractor, Network, ViT, heads
from memorax.utils.wrappers import PufferLibWrapper

total_timesteps = 50_000_000
min_steps_per_env = 10_000  # Minimum steps per env per training call

seed = 0
num_envs = 8  # Number of parallel environments


def make(variants=None, **kwargs):
    cfg = CogsGuardBasicMission.make_env()

    if variants:
        apply_reward_variants(cfg, variants=variants)

    simulator = Simulator()
    return MettaGridPufferEnv(simulator, cfg, **kwargs)


puffer_env = pufferlib.vector.make(
    partial(make, variants=["credit"]),
    num_envs=num_envs,
    backend=pufferlib.vector.Serial,
)

env = PufferLibWrapper(puffer_env)
env_params = env.default_params


cfg = PPOConfig(
    num_envs=env.num_envs,  # Use num_envs from the env as this is num_envs x num_agents
    num_steps=128,
    gae_lambda=0.95,
    num_minibatches=4,
    update_epochs=4,
    normalize_advantage=True,
    clip_coefficient=0.2,
    clip_value_loss=True,
    entropy_coefficient=0.01,
)

num_train_steps = 10_000 * env.num_envs

feature_extractor = FeatureExtractor(
    observation_extractor=nn.Sequential(
        (
            ViT(features=128, num_layers=2, num_heads=4),
            nn.Dense(128, kernel_init=nn.initializers.orthogonal(scale=1.414)),
        )
    ),
)
actor_network = Network(
    feature_extractor=feature_extractor,
    head=heads.Categorical(
        action_dim=env.num_actions,
        kernel_init=nn.initializers.orthogonal(scale=0.01),
    ),
)

critic_network = Network(
    feature_extractor=feature_extractor,
    head=heads.VNetwork(
        kernel_init=nn.initializers.orthogonal(scale=1.0),
    ),
)

optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adam(learning_rate=3e-4, eps=1e-5),
)


key = jax.random.key(seed)

agent = PPO(
    cfg=cfg,
    env=env,
    env_params=env_params,
    actor_network=actor_network,
    critic_network=critic_network,
    actor_optimizer=optimizer,
    critic_optimizer=optimizer,
)

logger = Logger(
    [DashboardLogger(title="PPO PufferLib CartPole", total_timesteps=total_timesteps)]
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
    info = jax.tree.map(lambda x: x.mean(), transitions.metadata)
    info = jax.tree.map(lambda x: x[None], info)
    data = {"training/SPS": SPS, **training_statistics, **info}
    logger_state = logger.log(logger_state, data, step=state.step.item())
    logger.emit(logger_state)
logger.finish(logger_state)
