import time
from dataclasses import asdict

import distrax
import flax.linen as nn
import jax
import optax

from memorax.algorithms import PPO, PPOConfig
from memorax.environments import environment
from memorax.loggers import DashboardLogger, Logger
from memorax.networks import FeatureExtractor, Identity, Network, heads
from memorax.utils.wrappers import (
    ClipActionWrapper,
    NormalizeObservationWrapper,
    ScaleRewardWrapper,
)

total_timesteps = 50_000_000
num_envs = 4096
max_steps_in_episode = 1000
num_train_steps = num_envs * max_steps_in_episode
num_eval_steps = 0

seed = 0
num_seeds = 1

env, env_params = environment.make("brax::hopper", mode="F")
env = NormalizeObservationWrapper(env)
env = ClipActionWrapper(env)
env = ScaleRewardWrapper(env, scale=10.0)

cfg = PPOConfig(
    name="PPO",
    num_envs=num_envs,
    num_steps=16,  # num_envs * num_steps = 65,536 total samples (matches Brax batch_size * num_minibatches)
    gae_lambda=0.95,
    num_minibatches=32,
    update_epochs=4,
    normalize_advantage=True,
    clip_coefficient=0.3,
    clip_value_loss=True,
    entropy_coefficient=0.01,
)

# Brax policy network: 4 layers of 32 with swish
actor_network = Network(
    feature_extractor=FeatureExtractor(
        observation_extractor=nn.Sequential(
            (
                nn.Dense(32, kernel_init=nn.initializers.lecun_uniform()),
                nn.swish,
                nn.Dense(32, kernel_init=nn.initializers.lecun_uniform()),
                nn.swish,
                nn.Dense(32, kernel_init=nn.initializers.lecun_uniform()),
                nn.swish,
                nn.Dense(32, kernel_init=nn.initializers.lecun_uniform()),
            )
        ),
    ),
    torso=Identity(),
    head=heads.Gaussian(
        action_dim=env.action_space(env_params).shape[0],
        kernel_init=nn.initializers.lecun_uniform(),
        transform=distrax.Tanh(),  # Squash actions to [-1, 1] like Brax
    ),
)

# Brax value network: 5 layers of 256 with swish
critic_network = Network(
    feature_extractor=FeatureExtractor(
        observation_extractor=nn.Sequential(
            (
                nn.Dense(256, kernel_init=nn.initializers.lecun_uniform()),
                nn.swish,
                nn.Dense(256, kernel_init=nn.initializers.lecun_uniform()),
                nn.swish,
                nn.Dense(256, kernel_init=nn.initializers.lecun_uniform()),
                nn.swish,
                nn.Dense(256, kernel_init=nn.initializers.lecun_uniform()),
                nn.swish,
                nn.Dense(256, kernel_init=nn.initializers.lecun_uniform()),
            )
        ),
    ),
    torso=Identity(),
    head=heads.VNetwork(gamma=0.97, kernel_init=nn.initializers.lecun_uniform()),
)

optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adam(learning_rate=3e-4, eps=1e-5),
)

key = jax.random.key(seed)
keys = jax.random.split(key, num_seeds)

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
    {
        "dashboard": DashboardLogger(
            title="PPO brax Example", total_timesteps=total_timesteps
        ),
    }
)
logger_state = logger.init(cfg=asdict(cfg))

init = jax.vmap(agent.init)
evaluate = jax.vmap(agent.evaluate, in_axes=(0, 0, None))
train = jax.vmap(agent.train, in_axes=(0, 0, None))

keys, state = init(keys)

keys, transitions = evaluate(keys, state, max_steps_in_episode)
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

    SPS = int(num_train_steps / (end - start))

    training_statistics = jax.vmap(Logger.get_episode_statistics, in_axes=(0, None))(
        transitions, "training"
    )
    losses = jax.vmap(
        lambda transition: jax.tree.map(lambda x: x.mean(), transition.losses)
    )(transitions)
    infos = jax.vmap(lambda t: t.infos)(transitions)
    data = {"training/SPS": SPS, **training_statistics, **losses, **infos}
    logger_state = logger.log(logger_state, data, step=state.step[0].item())

    if num_eval_steps > 0:
        keys, transitions = evaluate(keys, state, num_eval_steps)
        evaluation_statistics = jax.vmap(
            Logger.get_episode_statistics, in_axes=(0, None)
        )(transitions, "evaluation")
        logger_state = logger.log(
            logger_state, evaluation_statistics, step=state.step[0].item()
        )
    logger.emit(logger_state)
logger.finish(logger_state)
