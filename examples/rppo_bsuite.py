import time

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
    RNN,
    xLSTMCell,
)

total_timesteps = 1_000_000
num_train_steps = 50_000
num_eval_steps = 10_000

seed = 0
num_seeds = 5

# env, env_params = environment.make("gymnax::CartPole-v1")
env, env_params = environment.make("gymnax::MemoryChain-bsuite")

memory_length = 127
env_params = env_params.replace(
    memory_length=memory_length, max_steps_in_episode=memory_length + 1
)


cfg = RPPOConfig(
    name="rppo",
    learning_rate=3e-4,
    num_envs=32,
    num_eval_envs=16,
    num_steps=256,
    anneal_lr=True,
    gamma=0.999,
    gae_lambda=0.95,
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
            features=(112,), kernel_init=nn.initializers.orthogonal(scale=1.414)
        ),
        action_extractor=MLP(
            features=(16,), kernel_init=nn.initializers.orthogonal(scale=1.414)
        ),
        action_extractor=MLP(
            features=(32,), kernel_init=nn.initializers.orthogonal(scale=1.414)
        ),
        reward_extractor=MLP(
            features=(32,), kernel_init=nn.initializers.orthogonal(scale=1.414)
        ),
    ),
    # torso=GPT2(
    #     features=256,
    #     num_embeddings=env_params.max_steps_in_episode,
    #     num_layers=4,
    #     num_heads=4,
    #     context_length=256,
    # ),
    # torso=GTrXL(
    #     features=256,
    #     num_layers=4,
    #     num_heads=4,
    #     context_length=64,
    #     memory_length=192,
    # ),
    # torso=S5(features=128, state_size=256, num_layers=4),
    # torso=FFM(
    #     features=128,
    #     memory_size=32,
    #     context_size=4,
    # ),
    # torso=RNN(cell=xLSTMCell(features=256, pattern=("s"))),
    torso=RNN(cell=nn.GRUCell(features=128)),
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
    feature_extractor=FeatureExtractor(
        observation_extractor=MLP(
            features=(112,), kernel_init=nn.initializers.orthogonal(scale=1.414)
        ),
        action_extractor=MLP(
            features=(16,), kernel_init=nn.initializers.orthogonal(scale=1.414)
        ),
        action_extractor=MLP(
            features=(32,), kernel_init=nn.initializers.orthogonal(scale=1.414)
        ),
        reward_extractor=MLP(
            features=(32,), kernel_init=nn.initializers.orthogonal(scale=1.414)
        ),
    ),
    # torso=GPT2(
    #     features=256,
    #     num_embeddings=env_params.max_steps_in_episode,
    #     num_layers=4,
    #     num_heads=4,
    #     context_length=256,
    # ),
    # torso=GTrXL(
    #     features=256,
    #     num_layers=4,
    #     num_heads=4,
    #     context_length=64,
    #     memory_length=192,
    # ),
    # torso=S5(features=128, state_size=256, num_layers=4),
    # torso=FFM(
    #     features=128,
    #     memory_size=32,
    #     context_size=4,
    # ),
    # torso=RNN(cell=xLSTMCell(features=256, pattern=("s"))),
    torso=RNN(cell=nn.GRUCell(features=128)),
    head=heads.VNetwork(),
)
critic_optimizer = optax.chain(
    optax.clip_by_global_norm(cfg.max_grad_norm),
    optax.adam(learning_rate=cfg.learning_rate, eps=1e-5),
)

key = jax.random.key(seed)
keys = jax.random.split(key, num_seeds)

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
        DashboardLogger(title="RPPO bsuite Example", total_timesteps=total_timesteps),
        WandbLogger(entity="noahfarr", project="benchmarks", mode="online", group=f"rppo_bsuite_{memory_length}_gpt2", num_seeds=num_seeds),
     ]
)
logger_state = logger.init(cfg)

init = jax.vmap(agent.init)
evaluate = jax.vmap(agent.evaluate, in_axes=(0, 0, None))
train = jax.vmap(agent.train, in_axes=(0, 0, None))

keys, state = init(keys)

keys, transitions = evaluate(keys, state, num_eval_steps)
evaluation_statistics = jax.vmap(
    Logger.get_episode_statistics, in_axes=(0, None)
)(transitions, "evaluation")
logger_state = logger.log(logger_state, evaluation_statistics, step=state.step[0].item())
logger.emit(logger_state)

for i in range(0, total_timesteps, num_train_steps):

    start = time.perf_counter()
    keys, state, transitions = train(keys, state, num_train_steps)
    jax.block_until_ready(state)
    end = time.perf_counter()

    SPS = int(num_train_steps / (end - start))

    training_statistics = jax.vmap(
        Logger.get_episode_statistics, in_axes=(0, None)
    )(transitions, "training")
    losses = jax.vmap(lambda transition: jax.tree.map(lambda x: x.mean(), transition.losses))(transitions)
    data = {"SPS": SPS, **training_statistics, **losses}
    logger_state = logger.log(logger_state, data, step=state.step[0].item())

    keys, transitions = evaluate(keys, state, num_eval_steps)
    evaluation_statistics = jax.vmap(
        Logger.get_episode_statistics, in_axes=(0, None)
    )(transitions, "evaluation")
    logger_state = logger.log(logger_state, evaluation_statistics, step=state.step[0].item())
    logger.emit(logger_state)
