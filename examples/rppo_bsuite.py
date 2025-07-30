import time

import jax
import flax.linen as nn
import optax
from memory_rl.algorithms.rppo import RPPO, RPPOConfig
from memory_rl.environments import environment
from memory_rl.loggers import Logger, DashboardLogger
from memory_rl.networks import (
    MLP,
    SequenceNetwork,
    heads,
    SharedFeatureExtractor,
    S5,
    RNN,
    SHMCell,
    FFM,
    GPT2,
    GTrXL,
    xLSTMCell,
)

total_timesteps = 5_000_000
num_train_steps = 50_000
num_eval_steps = 10_000

# env, env_params = environment.make("gymnax::CartPole-v1")
env, env_params = environment.make("gymnax::MemoryChain-bsuite")

memory_length = 511
env_params = env_params.replace(
    memory_length=memory_length, max_steps_in_episode=memory_length + 1
)


cfg = RPPOConfig(
    name="rppo",
    learning_rate=3e-4,
    num_envs=32,
    num_eval_envs=16,
    num_steps=512,
    anneal_lr=True,
    gamma=0.999,
    gae_lambda=0.95,
    num_minibatches=8,
    update_epochs=4,
    normalize_advantage=True,
    clip_coef=0.2,
    clip_vloss=True,
    ent_coef=0.0,
    vf_coef=0.5,
    max_grad_norm=0.5,
    learning_starts=0,
)

actor_network = SequenceNetwork(
    feature_extractor=SharedFeatureExtractor(extractor=MLP(features=(128,))),
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
    # torso=S5(features=128, state_size=256, num_layers=4),
    # torso=FFM(
    #     features=128,
    #     memory_size=32,
    #     context_size=4,
    # ),
    torso=RNN(cell=xLSTMCell(features=128, pattern=("m", "s"))),
    head=heads.Categorical(
        action_dim=env.action_space(env_params).n,
    ),
)
actor_optimizer = optax.chain(
    optax.clip_by_global_norm(cfg.max_grad_norm),
    optax.adam(learning_rate=cfg.learning_rate, eps=1e-5),
    # optax.contrib.muon(learning_rate=cfg.learning_rate),
)

critic_network = SequenceNetwork(
    feature_extractor=SharedFeatureExtractor(extractor=MLP(features=(128,))),
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
    # torso=S5(features=128, state_size=256, num_layers=4),
    # torso=FFM(
    #     features=128,
    #     memory_size=32,
    #     context_size=4,
    # ),
    torso=RNN(cell=xLSTMCell(features=128, pattern=("m", "s"))),
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
