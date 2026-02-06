import time
from dataclasses import asdict
from functools import partial

import flax.linen as nn
import jax

# GPU optimization: Use TensorFloat32 for faster matmuls on modern NVIDIA GPUs (L4, A100, etc.)
jax.config.update("jax_default_matmul_precision", "tensorfloat32")
# Disable PGLE (Profile Guided Latency Estimator) which uses CUDA graph capture
# This is required because pure_callback and scatter operations don't support graph capture
jax.config.update("jax_pgle_profiling_runs", 0)
import optax
import pufferlib
from cogames.cli.mission import get_mission
from cogames.cogs_vs_clips.cogsguard_reward_variants import apply_reward_variants
from mettagrid.envs.mettagrid_puffer_env import MettaGridPufferEnv
from mettagrid.simulator import Simulator

from memorax.algorithms import PPO, PPOConfig
from memorax.loggers import DashboardLogger, Logger
from memorax.networks import MLP, FeatureExtractor, Network, TokenEmbedding, ViT, heads
from memorax.utils.wrappers import PufferLibWrapper


total_timesteps = 50_000_000
min_steps_per_env = 10_000  # Minimum steps per env per training call

seed = 0
num_envs = 64  # Number of parallel environments (increased for L4 GPU with 24GB VRAM)


def make(env_id, variants=None, **kwargs):
    _, cfg, _ = get_mission(env_id)

    if variants:
        apply_reward_variants(cfg, variants=variants)

    simulator = Simulator()
    return MettaGridPufferEnv(simulator, cfg, **kwargs)


simulator = Simulator()
puffer_env = pufferlib.vector.make(
    partial(make, variants=["credit"]),
    num_envs=num_envs,
    backend=pufferlib.vector.Serial,
    env_kwargs={"env_id": "cogsguard_arena.basic"},
)

env = PufferLibWrapper(puffer_env)
env_params = env.default_params


cfg = PPOConfig(
    name="PPO",
    num_envs=env.num_envs,  # Use num_envs from the env as this is num_envs x num_agents
    num_eval_envs=0,
    num_steps=64,
    gamma=0.99,
    gae_lambda=0.95,
    num_minibatches=16,  # Increased proportionally with num_envs (keeps minibatch ~256)
    update_epochs=4,
    normalize_advantage=True,
    clip_coef=0.2,
    clip_vloss=True,
    ent_coef=0.01,
    vf_coef=0.5,
)

num_train_steps = 10_000 * env.num_envs

d_model = 128
d_embed = 32

token_embedding = TokenEmbedding(features=d_embed, num_features=3, num_embeddings=256)
observation_extractor = ViT(
    features=d_model,
    num_layers=2,
    num_heads=4,
    expansion_factor=4,
    patch_embedding=token_embedding,
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
    info = jax.tree.map(lambda x: x.mean(), transitions.info)
    info = jax.tree.map(lambda x: x[None], info)
    info = {f"training/{k}": v for k, v in info.items()}
    data = {"training/SPS": SPS, **training_statistics, **info}
    logger_state = logger.log(logger_state, data, step=state.step.item())
    logger.emit(logger_state)
