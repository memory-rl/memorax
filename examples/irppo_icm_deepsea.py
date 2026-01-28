"""IRPPO with ICM on DeepSea.

This example demonstrates Intrinsic Reward PPO with the Intrinsic Curiosity Module (ICM)
on the BSuite DeepSea environment - a classic hard exploration benchmark.

DeepSea is an NxN grid where the agent starts at top-left and must reach bottom-right.
At each step, the agent chooses left or right. Only reaching the goal gives reward,
making random exploration exponentially unlikely to succeed (2^N probability).

ICM provides curiosity-driven intrinsic rewards based on forward model prediction error,
encouraging the agent to explore novel states rather than relying on sparse extrinsic rewards.

Key components:
- IRPPO: PPO variant that augments extrinsic rewards with intrinsic rewards
- ICM: Curiosity-driven exploration module with:
  - Encoder: Maps observations to feature space
  - Forward model: Predicts next features from (features, action)
  - Inverse model: Predicts action from (features, next_features)
"""

import time
from dataclasses import asdict

import flax.linen as nn
import jax
import optax

from gymnax.wrappers import FlattenObservationWrapper

from memorax.algorithms import IRPPO, IRPPOConfig
from memorax.environments import environment
from memorax.intrinsic_rewards import ICM
from memorax.loggers import DashboardLogger, Logger
from memorax.networks import MLP, FeatureExtractor, Network, SequenceModelWrapper, heads

total_timesteps = 500_000
num_train_steps = 10_000
num_eval_steps = 10_000

seed = 0
num_seeds = 1

env, env_params = environment.make("gymnax::DeepSea-bsuite")
env = FlattenObservationWrapper(env)  # Flatten 2D observations (8x8 -> 64)
num_actions = env.action_space(env_params).n
obs_shape = env.observation_space(env_params).shape

# IRPPO configuration with intrinsic reward coefficient
cfg = IRPPOConfig(
    name="IRPPO-ICM-DeepSea",
    num_envs=8,
    num_eval_envs=16,
    num_steps=128,
    gamma=0.99,
    gae_lambda=0.95,
    num_minibatches=4,
    update_epochs=4,
    normalize_advantage=True,
    clip_coef=0.2,
    clip_vloss=True,
    ent_coef=0.01,
    vf_coef=0.5,
    intrinsic_reward_coef=0.01,  # Weight for intrinsic rewards
)

# Actor-Critic networks
feature_extractor = FeatureExtractor(
    observation_extractor=MLP(
        features=(128,), kernel_init=nn.initializers.orthogonal(scale=1.414)
    ),
)
torso = SequenceModelWrapper(
    MLP(features=(128,), kernel_init=nn.initializers.orthogonal(scale=1.414))
)
actor_network = Network(
    feature_extractor=feature_extractor,
    torso=torso,
    head=heads.Categorical(
        action_dim=num_actions,
        kernel_init=nn.initializers.orthogonal(scale=0.01),
    ),
)
critic_network = Network(
    feature_extractor=feature_extractor,
    torso=torso,
    head=heads.VNetwork(
        kernel_init=nn.initializers.orthogonal(scale=1.0),
    ),
)

# ICM components
icm_feature_dim = 64

icm_encoder = MLP(
    features=(128, icm_feature_dim),
    kernel_init=nn.initializers.orthogonal(scale=1.414),
)
# Forward model: features + one-hot action -> next features
icm_forward_model = MLP(
    features=(128, icm_feature_dim),
    kernel_init=nn.initializers.orthogonal(scale=1.414),
)
# Inverse model: features + next_features -> action logits
icm_inverse_model = MLP(
    features=(128, num_actions),
    kernel_init=nn.initializers.orthogonal(scale=1.414),
)

icm = ICM(
    encoder=icm_encoder,
    forward_model=icm_forward_model,
    inverse_model=icm_inverse_model,
    num_actions=num_actions,
    beta=0.2,  # Weight for forward loss vs inverse loss
)

# Optimizers
optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adam(learning_rate=3e-4, eps=1e-5),
)
icm_optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adam(learning_rate=1e-3, eps=1e-5),
)

key = jax.random.key(seed)
keys = jax.random.split(key, num_seeds)

agent = IRPPO(
    cfg=cfg,
    env=env,
    env_params=env_params,
    actor_network=actor_network,
    critic_network=critic_network,
    actor_optimizer=optimizer,
    critic_optimizer=optimizer,
    intrinsic_reward_network=icm,
    intrinsic_reward_optimizer=icm_optimizer,
)

logger = Logger(
    [DashboardLogger(title="IRPPO-ICM DeepSea", total_timesteps=total_timesteps)]
)
logger_state = logger.init(cfg=asdict(cfg))

init = jax.vmap(agent.init)
evaluate = jax.vmap(agent.evaluate, in_axes=(0, 0, None))
train = jax.vmap(agent.train, in_axes=(0, 0, None))

keys, state = init(keys)

keys, transitions = evaluate(keys, state, num_eval_steps)
evaluation_statistics = jax.vmap(Logger.get_episode_statistics, in_axes=(0, None))(
    transitions, "evaluation"
)
logger_state = logger.log(
    logger_state, evaluation_statistics, step=state.step[0].item()
)
logger.emit(logger_state)

for i in range(0, total_timesteps, num_train_steps):
    start = time.perf_counter()
    keys, state, transitions = train(keys, state, num_train_steps)
    jax.block_until_ready(state)
    end = time.perf_counter()

    SPS = num_train_steps / (end - start)

    training_statistics = jax.vmap(Logger.get_episode_statistics, in_axes=(0, None))(
        transitions, "training"
    )
    losses = jax.vmap(
        lambda transition: jax.tree.map(lambda x: x.mean(), transition.losses)
    )(transitions)
    data = {"training/SPS": SPS, **training_statistics, **losses}
    logger_state = logger.log(logger_state, data, step=state.step[0].item())

    keys, transitions = evaluate(keys, state, num_eval_steps)
    evaluation_statistics = jax.vmap(Logger.get_episode_statistics, in_axes=(0, None))(
        transitions, "evaluation"
    )
    logger_state = logger.log(
        logger_state, evaluation_statistics, step=state.step[0].item()
    )
    logger.emit(logger_state)
