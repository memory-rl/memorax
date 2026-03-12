import time
from dataclasses import asdict

import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from gymnax.wrappers import LogWrapper

from memorax.algorithms.gradient_ppo import GradientPPO, GradientPPOConfig
from memorax.environments import environment
from memorax.environments.wrappers import ClipActionWrapper, NormalizeObservationWrapper
from memorax.loggers import DashboardLogger, Logger
from memorax.networks import FeatureExtractor, Identity, Network, heads

seed = 0
num_seeds = 1

env, env_params = environment.make("brax::ant")
env = LogWrapper(env)
env = NormalizeObservationWrapper(env)
env = ClipActionWrapper(env)

cfg = GradientPPOConfig(
    num_envs=64,
    num_steps=32,
    gae_lambda=0.95,
    num_minibatches=8,
    update_epochs=4,
    normalize_advantage=True,
    clip_coefficient=0.2,
    clip_value_loss=False,
    entropy_coefficient=0.0,
    regularization_coefficient=1.0,
    truncation_length=32,
)

total_timesteps = 5_000_000
num_train_steps = 1000 * cfg.num_envs * cfg.num_steps

feature_extractor = FeatureExtractor(
    observation_extractor=nn.Sequential(
        (
            nn.Dense(64, kernel_init=nn.initializers.orthogonal(scale=1.414)),
            nn.tanh,
            nn.Dense(64, kernel_init=nn.initializers.orthogonal(scale=1.414)),
            nn.tanh,
        )
    ),
)

actor_network = Network(
    feature_extractor=feature_extractor,
    torso=Identity(),
    head=heads.Gaussian(
        action_dim=env.action_space(env_params).shape[0],
        kernel_init=nn.initializers.orthogonal(scale=0.01),
    ),
)

critic_network = Network(
    feature_extractor=feature_extractor,
    torso=Identity(),
    head=heads.VNetwork(
        gamma=0.99,
        kernel_init=nn.initializers.orthogonal(scale=1.0),
    ),
)

h_network = Network(
    feature_extractor=feature_extractor,
    torso=Identity(),
    head=heads.VNetwork(
        gamma=0.99,
        kernel_init=nn.initializers.orthogonal(scale=1.0),
    ),
)

actor_optimizer = optax.chain(
    optax.clip_by_global_norm(0.5),
    optax.adam(learning_rate=3e-4, eps=1e-5),
)
critic_optimizer = optax.chain(
    optax.clip_by_global_norm(0.5),
    optax.adam(learning_rate=3e-3, eps=1e-5),
)
h_optimizer = optax.chain(
    optax.clip_by_global_norm(0.5),
    optax.adam(learning_rate=3e-3, eps=1e-5),
)

key = jax.random.key(seed)
keys = jax.random.split(key, num_seeds)

agent = GradientPPO(
    cfg=cfg,
    env=env,
    env_params=env_params,
    actor_network=actor_network,
    critic_network=critic_network,
    h_network=h_network,
    actor_optimizer=actor_optimizer,
    critic_optimizer=critic_optimizer,
    h_optimizer=h_optimizer,
)

logger = Logger(
    [
        DashboardLogger(
            title="Gradient PPO Ant",
            total_timesteps=total_timesteps,
            env_id="Brax Ant",
        )
    ]
)
logger_state = logger.init(cfg=asdict(cfg))

init = jax.vmap(agent.init)
train = jax.vmap(agent.train, in_axes=(0, 0, None))

keys, state = init(keys)

for i in range(0, total_timesteps, num_train_steps):
    start = time.perf_counter()
    keys, state, transitions = train(keys, state, num_train_steps)
    jax.block_until_ready(state)
    end = time.perf_counter()

    SPS = int(num_train_steps / (end - start))

    losses = jax.vmap(
        lambda transition: jax.tree.map(lambda x: x.mean(), transition.losses)
    )(transitions)

    def episode_statistics(t):
        mask = t.info["returned_episode"]
        returns = jnp.where(mask, t.info["returned_episode_returns"], jnp.nan)
        lengths = jnp.where(mask, t.info["returned_episode_lengths"], jnp.nan)
        return {
            "training/num_episodes": mask.sum(),
            "training/mean_episode_returns": jnp.nanmean(returns),
            "training/std_episode_returns": jnp.nanstd(returns),
            "training/mean_episode_lengths": jnp.nanmean(lengths),
            "training/std_episode_lengths": jnp.nanstd(lengths),
        }

    training_statistics = jax.vmap(episode_statistics)(transitions)
    data = {"training/SPS": SPS, **training_statistics, **losses}
    logger_state = logger.log(logger_state, data, step=state.step[0].item())
    logger.emit(logger_state)

logger.finish(logger_state)
