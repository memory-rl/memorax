import time

import flax.linen as nn
import jax
import optax

from memorax.algorithms import PQN, PQNConfig
from memorax.environments import environment
from memorax.loggers import DashboardLogger, Logger, WandbLogger
from memorax.networks import (GPT2, LRU, MLP, RNN, DeltaNet, DeltaProduct,
                              FeatureExtractor, GatedDeltaNet, GTrXL, Mamba,
                              Network, RecurrentWrapper, TDDeltaNet,
                              TDGatedDeltaNet, heads, xLSTMCell)

total_timesteps = 1_000_000
num_train_steps = 100_000
num_eval_steps = 10_000

seed = 0
num_seeds = 1

env, env_params = environment.make("gymnax::CartPole-v1")
# env, env_params = environment.make("gymnax::MemoryChain-bsuite")

# memory_length = 5
# env_params = env_params.replace(
#     memory_length=memory_length, max_steps_in_episode=memory_length + 1
# )


cfg = PQNConfig(
    name="PQN",
    learning_rate=1e-3,
    num_envs=64,
    num_eval_envs=16,
    num_steps=32,
    gamma=0.99,
    td_lambda=0.95,
    num_minibatches=8,
    update_epochs=4,
    start_e=1.0,
    end_e=0.05,
    exploration_fraction=0.5,
    max_grad_norm=1.0,
    learning_starts=0,
    shuffle_time_axis=True,
)

q_network = Network(
    feature_extractor=FeatureExtractor(
        observation_extractor=MLP(
            features=(192,), kernel_init=nn.initializers.orthogonal(scale=1.414)
        ),
    ),
    torso=RNN(cell=nn.GRUCell(features=256)),
    # torso=RecurrentWrapper(
    #     MLP(features=(256,), kernel_init=nn.initializers.orthogonal(scale=1.414)),
    # ),
    head=heads.DiscreteQNetwork(
        action_dim=env.action_space(env_params).n,
    ),
)
optimizer = optax.chain(
    optax.clip_by_global_norm(cfg.max_grad_norm),
    optax.adam(learning_rate=cfg.learning_rate, eps=1e-5),
)

schedule = optax.linear_schedule(
    cfg.start_e, cfg.end_e, cfg.exploration_fraction * total_timesteps
)

key = jax.random.key(seed)
keys = jax.random.split(key, num_seeds)

agent = PQN(
    cfg=cfg,
    env=env,
    env_params=env_params,
    q_network=q_network,
    optimizer=optimizer,
    epsilon_schedule=schedule,
)

logger = Logger(
    [
        DashboardLogger(title="PQN bsuite Example", total_timesteps=total_timesteps),
        # WandbLogger(entity="noahfarr", project="memorax", name="PQN_bsuite"),
    ]
)
logger_state = logger.init(cfg)

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
