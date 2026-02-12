import time
from dataclasses import asdict

import flax.linen as nn
import jax
import jaxmarl
import optax

from memorax.algorithms import IPPO, IPPOConfig
from memorax.environments.jaxmarl import JaxMarlWrapper
from memorax.loggers import DashboardLogger, Logger
from memorax.networks import FeatureExtractor, Network, heads

total_timesteps = 1_000_000
num_train_steps = 10_000
num_eval_steps = 10_000

seed = 0
num_seeds = 1

base_env = jaxmarl.make("MPE_simple_spread_v3")
env = JaxMarlWrapper(base_env)

cfg = IPPOConfig(
    num_envs=8,
    num_eval_envs=16,
    num_steps=128,
    gamma=0.99,
    gae_lambda=0.95,
    num_minibatches=4,
    update_epochs=4,
    normalize_advantage=True,
    clip_coef=0.2,
    clip_vloss=True,
    ent_coef=0.01,
    vf_coef=0.5,
)

d_model = 128

feature_extractor = FeatureExtractor(
    observation_extractor=nn.Sequential((
        nn.Dense(d_model, kernel_init=nn.initializers.orthogonal(scale=1.414)), nn.relu,
        nn.Dense(d_model, kernel_init=nn.initializers.orthogonal(scale=1.414)), nn.relu,
    )),
)

action_space = env.action_spaces[env.agents[0]]

VmappedNetwork = nn.vmap(
    Network,
    variable_axes={"params": None},
    split_rngs={"params": False, "memory": True, "dropout": True},
    in_axes=(0, 0, 0, 0, 0, 0),
    out_axes=(0, 0),
)

actor_network = VmappedNetwork(
    feature_extractor=feature_extractor,

    head=heads.Categorical(
        action_dim=action_space.n,
        kernel_init=nn.initializers.orthogonal(scale=0.01),
    ),
)

critic_network = VmappedNetwork(
    feature_extractor=feature_extractor,

    head=heads.VNetwork(
        kernel_init=nn.initializers.orthogonal(scale=1.0),
    ),
)

optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adam(learning_rate=3e-4, eps=1e-5),
)


key = jax.random.key(seed)
keys = jax.random.split(key, num_seeds)

agent = IPPO(
    cfg=cfg,
    env=env,
    actor_network=actor_network,
    critic_network=critic_network,
    actor_optimizer=optimizer,
    critic_optimizer=optimizer,
)

logger = Logger(
    [DashboardLogger(title="IPPO MPE Simple Spread", total_timesteps=total_timesteps, name="IPPO", env_id="MPE_simple_spread_v3")]
)
logger_state = logger.init(cfg=asdict(cfg))

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
    infos = jax.vmap(lambda t: t.infos)(transitions)
    data = {"training/SPS": SPS, **training_statistics, **losses, **infos}
    logger_state = logger.log(logger_state, data, step=state.step[0].item())

    keys, transitions = evaluate(keys, state, num_eval_steps)
    evaluation_statistics = jax.vmap(Logger.get_episode_statistics, in_axes=(0, None))(
        transitions, "evaluation"
    )
    logger_state = logger.log(
        logger_state, evaluation_statistics, step=state.step[0].item()
    )
    logger.emit(logger_state)
logger.finish(logger_state)
