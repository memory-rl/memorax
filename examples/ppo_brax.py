import time

import flax.linen as nn
import jax
import jax.numpy as jnp
import optax

from memorax.algorithms import PPO, PPOConfig
from memorax.environments import environment
from memorax.loggers import DashboardLogger, Logger, WandbLogger
from memorax.networks import MLP, RNN, FeatureExtractor, Network, heads
from memorax.networks.sequence_models.wrappers import SequenceModelWrapper

total_timesteps = 50_000_000
num_train_steps = 4_096_000
num_eval_steps = 1_000

seed = 0
num_seeds = 1

env, env_params = environment.make("brax::ant", mode="F")

cfg = PPOConfig(
    name="PPO",
    num_envs=4096,
    num_eval_envs=4096,
    num_steps=5,
    gamma=0.97,
    gae_lambda=0.95,
    num_minibatches=32,
    update_epochs=4,
    normalize_advantage=True,
    clip_coef=0.3,
    clip_vloss=True,
    ent_coef=0.01,
    vf_coef=0.5,
)

feature_extractor = FeatureExtractor(
    observation_extractor=MLP(
        features=(128,), kernel_init=nn.initializers.orthogonal(scale=1.414)
    ),
)
torso = SequenceModelWrapper(
    network=MLP(features=(128,), kernel_init=nn.initializers.orthogonal(scale=1.414))
)
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
    [
        DashboardLogger(title="PPO brax Example", total_timesteps=total_timesteps),
    ]
)
logger_state = logger.init(cfg=cfg)

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

    SPS = int(num_train_steps / (end - start))

    training_statistics = jax.vmap(Logger.get_episode_statistics, in_axes=(0, None))(
        transitions, "training"
    )
    losses = jax.vmap(
        lambda transition: jax.tree.map(lambda x: x.mean(), transition.losses)
    )(transitions)
    data = {"SPS": SPS, **training_statistics, **losses}
    logger_state = logger.log(logger_state, data, step=state.step[0].item())

    keys, transitions = evaluate(keys, state, num_eval_steps)
    evaluation_statistics = jax.vmap(Logger.get_episode_statistics, in_axes=(0, None))(
        transitions, "evaluation"
    )
    logger_state = logger.log(
        logger_state, evaluation_statistics, step=state.step[0].item()
    )
    logger.emit(logger_state)
