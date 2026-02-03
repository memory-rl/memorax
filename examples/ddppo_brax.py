import time
from dataclasses import asdict

import flax.linen as nn
import jax
import optax

from memorax.algorithms import DDPPO, DDPPOConfig
from memorax.environments import environment
from memorax.loggers import DashboardLogger, Logger
from memorax.networks import MLP, FeatureExtractor, Network, heads
from memorax.utils.wrappers import (
    ClipActionWrapper,
    NormalizeObservationWrapper,
    ScaleRewardWrapper,
)

total_timesteps = 50_000_000
num_envs = 1024 // jax.local_device_count()
max_steps_in_episode = 1000
num_train_steps = num_envs * max_steps_in_episode * jax.local_device_count()
num_eval_steps = 0

seed = 0

env, env_params = environment.make("brax::hopper", mode="F")
env = NormalizeObservationWrapper(env)
env = ClipActionWrapper(env)
env = ScaleRewardWrapper(env, scale=10.0)

cfg = DDPPOConfig(
    name="DDPPO",
    num_envs=num_envs,
    num_eval_envs=num_envs,
    num_steps=128,
    gamma=0.97,
    gae_lambda=0.95,
    num_minibatches=32,
    update_epochs=4,
    normalize_advantage=True,
    clip_coef=0.3,
    clip_vloss=True,
    ent_coef=0.01,
)

feature_extractor = FeatureExtractor(
    observation_extractor=MLP(
        features=(128,), kernel_init=nn.initializers.orthogonal(scale=1.414)
    ),
)
torso = MLP(features=(128,), kernel_init=nn.initializers.orthogonal(scale=1.414))
actor_network = Network(
    feature_extractor=feature_extractor,
    torso=torso,
    head=heads.Gaussian(
        action_dim=env.action_space(env_params).shape[0],
        kernel_init=nn.initializers.orthogonal(scale=0.01),
    ),
)

critic_network = Network(
    feature_extractor=feature_extractor,
    torso=torso,
    head=heads.VNetwork(),
)

optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adam(learning_rate=3e-4, eps=1e-5),
)

key = jax.random.key(seed)

agent = DDPPO(
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
            title="DDPPO brax Example", total_timesteps=total_timesteps
        ),
    }
)
logger_state = logger.init(cfg=asdict(cfg))

key, state = agent.init(key)
keys, state = agent.replicate(key, state)

keys, transitions = agent.evaluate(keys, state, max_steps_in_episode, True)
evaluation_statistics = jax.vmap(Logger.get_episode_statistics, in_axes=(0, None))(
    transitions, "evaluation"
)
logger_state = logger.log(
    logger_state, evaluation_statistics, step=state.step[0].item()
)
logger.emit(logger_state)

for i in range(0, total_timesteps, num_train_steps):
    start = time.perf_counter()
    keys, state, transitions = agent.train(keys, state, num_envs * max_steps_in_episode)
    jax.block_until_ready(state)
    end = time.perf_counter()

    SPS = int(num_train_steps / (end - start))

    training_statistics = jax.vmap(Logger.get_episode_statistics, in_axes=(0, None))(
        transitions, "training"
    )
    losses = jax.vmap(
        lambda transition: jax.tree.map(lambda x: x.mean(), transition.losses)
    )(transitions)
    data = {"training/SPS": SPS, **training_statistics, **losses}
    logger_state = logger.log(logger_state, data, step=state.step[0].item())

    if num_eval_steps > 0:
        keys, transitions = agent.evaluate(keys, state, num_eval_steps, True)
        evaluation_statistics = jax.vmap(
            Logger.get_episode_statistics, in_axes=(0, None)
        )(transitions, "evaluation")
        logger_state = logger.log(
            logger_state, evaluation_statistics, step=state.step[0].item()
        )
    logger.emit(logger_state)
