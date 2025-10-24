import time
from dataclasses import asdict

import jax
import flax.linen as nn
import optax
from memory_rl.algorithms.rppo import RPPO, RPPOConfig
from memory_rl.environments import environment
from memory_rl.loggers import Logger, DashboardLogger, WandbLogger
from memory_rl.networks import (
    MLP,
    SequenceNetwork,
    heads,
    FeatureExtractor,
    S5,
    SHMCell,
    FFM,
    GPT2,
    GTrXL,
    xLSTMCell,
)
from memory_rl.networks.recurrent.rnn import RNN

total_timesteps = 1_000_000_000
num_train_steps = 200_000
num_eval_steps = 5_000

env, env_params = environment.make("craftax::Craftax-Symbolic-v1", auto_reset=True)

cfg = RPPOConfig(
    name="rppo",
    learning_rate=2e-4,
    num_envs=512,
    num_eval_envs=10,
    num_steps=64,
    anneal_lr=True,
    gamma=0.99,
    gae_lambda=0.8,
    num_minibatches=8,
    update_epochs=4,
    normalize_advantage=True,
    clip_coef=0.2,
    clip_vloss=True,
    ent_coef=0.01,
    vf_coef=0.5,
    max_grad_norm=1.0,
    learning_starts=0,
)

actor_network = SequenceNetwork(
    feature_extractor=FeatureExtractor(
        observation_extractor=MLP(
            features=(192,), kernel_init=nn.initializers.orthogonal(scale=1.414)
        ),
        action_extractor=MLP(
            features=(64,), kernel_init=nn.initializers.orthogonal(scale=1.414)
        ),
    ),
    # torso=GTrXL(
    #     features=256, num_layers=2, num_heads=8, context_length=64, memory_length=128
    # ),
    # torso=S5(features=128, state_size=32, num_layers=4),
    # torso=FFM(features=128, memory_size=32, context_size=16),
    torso=RNN(cell=xLSTMCell(features=256, pattern=("m"))),
    head=heads.Categorical(
        action_dim=env.action_space(env_params).n,
    ),
)
actor_optimizer = optax.chain(
    optax.clip_by_global_norm(cfg.max_grad_norm),
    optax.adam(learning_rate=cfg.learning_rate, eps=1e-5),
)

critic_network = SequenceNetwork(
    feature_extractor=FeatureExtractor(
        observation_extractor=MLP(
            features=(192,), kernel_init=nn.initializers.orthogonal(scale=1.414)
        ),
        action_extractor=MLP(
            features=(64,), kernel_init=nn.initializers.orthogonal(scale=1.414)
        ),
    ),
    # torso=GTrXL(
    #     features=256, num_layers=2, num_heads=8, context_length=64, memory_length=128
    # ),
    # torso=GTrXL(
    #     features=128, num_layers=4, num_heads=4, context_length=64, memory_length=64
    # ),
    # torso=FFM(features=128, memory_size=32, context_size=16),
    torso=RNN(cell=xLSTMCell(features=256, pattern=("m"))),
    head=heads.VNetwork(),
)
critic_optimizer = optax.chain(
    optax.clip_by_global_norm(cfg.max_grad_norm),
    optax.adam(learning_rate=cfg.learning_rate, eps=1e-5),
)

key = jax.random.key(1)

agent = RPPO(
    cfg=cfg,
    env=env,
    env_params=env_params,
    actor=actor_network,
    critic=critic_network,
    actor_optimizer=actor_optimizer,
    critic_optimizer=critic_optimizer,
)

logger = Logger(
    [
        DashboardLogger(title="RPPO Craftax Example", name="RPPO", env_id="Craftax", total_timesteps=total_timesteps),
    ]
)
logger_state = logger.init(asdict(cfg))

key, state = agent.init(key)

key, transitions = agent.evaluate(key, state, num_steps=num_eval_steps)
evaluation_statistics = Logger.get_episode_statistics(transitions, "evaluation")
logger_state = logger.log(logger_state, evaluation_statistics, step=state.step.item())
logger.emit(logger_state)

for i in range(0, total_timesteps, num_train_steps):

    start = time.perf_counter()
    key, state, transitions = agent.train(key, state, num_steps=num_train_steps)
    jax.block_until_ready(state)
    end = time.perf_counter()

    SPS = int(num_train_steps / (end - start))

    training_statistics = Logger.get_episode_statistics(transitions, "training")
    data = {"SPS": SPS, **training_statistics, **transitions.losses}
    logger_state = logger.log(logger_state, data, step=state.step.item())

    key, transitions = agent.evaluate(key, state, num_steps=num_eval_steps)
    evaluation_statistics = Logger.get_episode_statistics(transitions, "evaluation")
    logger_state = logger.log(
        logger_state, evaluation_statistics, step=state.step.item()
    )
    logger.emit(logger_state)
