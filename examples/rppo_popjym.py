import time

import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from memorax.algorithms.rppo import RPPO, RPPOConfig
from memorax.environments import environment
from memorax.loggers import Logger, DashboardLogger
from memorax.networks import (
    MLP,
    SequenceNetwork,
    heads,
    FeatureExtractor,
    RNN,
    SHMCell,
    FFM,
    GPT2,
    GTrXL,
    xLSTMCell,
)

total_timesteps = 15_000_000
num_train_steps = 1024 * 64
num_eval_steps = 10_000

env, env_params = environment.make("popjym::RepeatFirstEasy")

cfg = RPPOConfig(
    name="rppo",
    learning_rate=5e-5,
    num_envs=64,
    num_eval_envs=10,
    num_steps=1024,
    anneal_lr=False,
    gamma=0.99,
    gae_lambda=1.0,
    num_minibatches=8,
    update_epochs=30,
    normalize_advantage=True,
    clip_coef=0.3,
    clip_vloss=True,
    ent_coef=0.0,
    vf_coef=1.0,
    max_grad_norm=1.0,
    learning_starts=0,
)

actor_network = SequenceNetwork(
    feature_extractor=FeatureExtractor(
        observation_extractor=MLP(features=(192,)),
        action_extractor=MLP(features=(64,)),
    ),
    # torso=GPT2(
    #     features=128,
    #     num_embeddings=env_params.max_steps_in_episode,
    #     num_layers=4,
    #     num_heads=4,
    #     context_length=128,
    # ),
    # torso=GTrXL(
    #     features=128,
    #     num_layers=4,
    #     num_heads=4,
    #     context_length=128,
    #     memory_length=1,
    # ),
    # torso=S5(features=128, state_size=32, num_layers=4),
    # torso=FFM(features=128, memory_size=32, context_size=8, num_steps=cfg.num_steps),
    # torso=RNN(cell=xLSTMCell(features=256, pattern=("m", "s"))),
    torso=RNN(cell=nn.GRUCell(features=128), unroll=16),
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
        observation_extractor=MLP(features=(192,)),
        action_extractor=MLP(features=(64,)),
    ),
    # torso=GPT2(
    #     features=128,
    #     num_embeddings=env_params.max_steps_in_episode,
    #     num_layers=4,
    #     num_heads=4,
    #     context_length=128,
    # ),
    # torso=GTrXL(
    #     features=128,
    #     num_layers=4,
    #     num_heads=4,
    #     context_length=256,
    #     memory_length=1,
    # ),
    # torso=FFM(features=128, memory_size=32, context_size=8, num_steps=cfg.num_steps),
    # torso=RNN(cell=xLSTMCell(features=256, pattern=("m", "s"))),
    torso=RNN(cell=nn.GRUCell(features=128), unroll=16),
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
    [DashboardLogger(title="RPPO bsuite Example", total_timesteps=total_timesteps)]
)
logger_state = logger.init(cfg)

key, state = agent.init(key)

<<<<<<< Updated upstream
mmer_evaluation = -jnp.inf
mmer_training = -jnp.inf

key, transitions = agent.evaluate(key, state, num_steps=num_eval_steps)
evaluation_statistics = Logger.get_episode_statistics(transitions, "evaluation")
mmer_evaluation = jnp.maximum(
    mmer_evaluation, evaluation_statistics["evaluation/mean_episode_returns"]
)
data = {**evaluation_statistics, "evaluation/mmer": mmer_evaluation}
logger_state = logger.log(logger_state, data, step=state.step.item())
=======
evaluation_mmer = -jnp.inf
traning_mmer = -jnp.inf

key, transitions = agent.evaluate(key, state, num_steps=num_eval_steps)
evaluation_statistics = Logger.get_episode_statistics(transitions, "evaluation")
evaluation_mmer = jnp.maximum(evaluation_mmer, evaluation_statistics["evaluation/mean_episode_returns"])
logger_state = logger.log(logger_state, {**evaluation_statistics, "evaluation/MMER": evaluation_mmer}, step=state.step.item())
>>>>>>> Stashed changes
logger.emit(logger_state)

for i in range(0, total_timesteps, num_train_steps):

    start = time.perf_counter()
    key, state, transitions = agent.train(key, state, num_steps=num_train_steps)
    jax.block_until_ready(state)
    end = time.perf_counter()

    SPS = int(num_train_steps / (end - start))

    training_statistics = Logger.get_episode_statistics(transitions, "training")
<<<<<<< Updated upstream
    mmer_training = jnp.maximum(
        mmer_training, training_statistics["training/mean_episode_returns"]
    )
    data = {
        "SPS": SPS,
        **training_statistics,
        **transitions.losses,
        "training/mmer": mmer_training,
    }
=======
    traning_mmer = jnp.maximum(traning_mmer, training_statistics["training/mean_episode_returns"])
    data = {"SPS": SPS, **training_statistics, **transitions.losses, "training/MMER": traning_mmer}
>>>>>>> Stashed changes
    logger_state = logger.log(logger_state, data, step=state.step.item())

    key, transitions = agent.evaluate(key, state, num_steps=num_eval_steps)
    evaluation_statistics = Logger.get_episode_statistics(transitions, "evaluation")
<<<<<<< Updated upstream
    mmer_evaluation = jnp.maximum(
        mmer_evaluation, evaluation_statistics["evaluation/mean_episode_returns"]
=======
    evaluation_mmer = jnp.maximum(evaluation_mmer, evaluation_statistics["evaluation/mean_episode_returns"])
    logger_state = logger.log(
        logger_state, {**evaluation_statistics, "evaluation/MMER": evaluation_mmer}, step=state.step.item()
>>>>>>> Stashed changes
    )
    data = {**evaluation_statistics, "evaluation/mmer": mmer_evaluation}
    logger_state = logger.log(logger_state, data, step=state.step.item())
    logger.emit(logger_state)
