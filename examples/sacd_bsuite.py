import time

from flashbax import make_item_buffer
import jax
import flax.linen as nn
import optax
from memorax.algorithms import SACD, SACDConfig
from memorax.environments import environment
from memorax.loggers import Logger, DashboardLogger, WandbLogger
from memorax.networks import (
    MLP,
    Network,
    heads,
    FeatureExtractor,
    RNN,
    GPT2,
    GTrXL,
    LRU,
    DeltaNet,
    TDDeltaNet,
    GatedDeltaNet,
    TDGatedDeltaNet,
    DeltaProduct,
    xLSTMCell,
    Mamba,
)

total_timesteps = 1_000_000
num_train_steps = 10_000
num_eval_steps = 5_000

seed = 0
num_seeds = 1

# env, env_params = environment.make("gymnax::CartPole-v1")
env, env_params = environment.make("gymnax::MemoryChain-bsuite")

memory_length = 5
env_params = env_params.replace(
    memory_length=memory_length, max_steps_in_episode=memory_length + 1
)


cfg = SACDConfig(
    name="SACD",
    actor_lr=3e-4,
    critic_lr=3e-4,
    alpha_lr=3e-4,
    num_envs=10,
    num_eval_envs=10,
    buffer_size=10_000,
    gamma=0.99,
    tau=1.0,
    target_update_frequency=500,
    batch_size=128,
    initial_alpha=1.0,
    target_entropy_scale=1.0,
    learning_starts=10_000,
    mask=False,
    train_frequency=10,
)

actor_network = Network(
    feature_extractor=FeatureExtractor(
        observation_extractor=MLP(
            features=(192,), kernel_init=nn.initializers.orthogonal(scale=1.414)
        ),
    ),
    torso=RNN(cell=nn.GRUCell(features=256)),
    head=heads.Categorical(
        action_dim=env.action_space(env_params).n,
    ),
)
actor_optimizer = optax.chain(
    optax.adam(learning_rate=cfg.actor_lr, eps=1e-5),
)

critic_network = nn.vmap(
    Network,
    variable_axes={"params": 0},
    split_rngs={"params": True},
    in_axes=None,
    out_axes=0,
    axis_size=2,
)(
    feature_extractor=FeatureExtractor(
        observation_extractor=MLP(
            features=(192,), kernel_init=nn.initializers.orthogonal(scale=1.414)
        ),
    ),
    torso=RNN(cell=nn.GRUCell(features=256)),
    head=heads.DiscreteQNetwork(
        action_dim=env.action_space(env_params).n,
    ),
)

critic_optimizer = optax.chain(
    optax.adam(learning_rate=cfg.critic_lr, eps=1e-5),
)

alpha_network = heads.Alpha(
    initial_alpha=cfg.initial_alpha,
)

alpha_optimizer = optax.chain(
    optax.adam(learning_rate=cfg.alpha_lr, eps=1e-5),
)

buffer = make_item_buffer(
    max_length=cfg.buffer_size,
    min_length=cfg.batch_size,
    sample_batch_size=cfg.batch_size,
    add_sequences=True,
    add_batches=True,
)

key = jax.random.key(seed)
keys = jax.random.split(key, num_seeds)

agent = SACD(
    cfg=cfg,
    env=env,
    env_params=env_params,
    actor_network=actor_network,
    critic_network=critic_network,
    alpha_network=alpha_network,
    actor_optimizer=actor_optimizer,
    critic_optimizer=critic_optimizer,
    alpha_optimizer=alpha_optimizer,
    buffer=buffer,
)

logger = Logger(
    [
        DashboardLogger(title="SACD bsuite Example", total_timesteps=total_timesteps),
        # WandbLogger(entity="noahfarr", project="memorax", name="SACD_bsuite"),
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
