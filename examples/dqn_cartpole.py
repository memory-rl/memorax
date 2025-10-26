import jax
import optax
import flashbax as fbx
from memory_rl.algorithms.dqn import DQN, DQNConfig
from memory_rl.environments import environment
from memory_rl.loggers import Logger, DashboardLogger
from memory_rl.networks import MLP, Network, heads, FeatureExtractor

total_timesteps = 500_000
num_train_steps = 50_000
num_eval_steps = 5_000

env, env_params = environment.make("gymnax::CartPole-v1")

cfg = DQNConfig(
    name="rppo",
    learning_rate=3e-4,
    num_envs=10,
    num_eval_envs=10,
    buffer_size=10_000,
    gamma=0.99,
    tau=1.0,
    target_network_frequency=500,
    batch_size=64,
    start_e=1.0,
    end_e=0.05,
    exploration_fraction=0.5,
    learning_starts=10_000,
    train_frequency=10,
    double=False,
)

q_network = Network(
    feature_extractor=FeatureExtractor(observation_extractor=MLP(features=(120,))),
    torso=MLP(features=(84,)),
    head=heads.DiscreteQNetwork(
        action_dim=env.action_space(env_params).n,
    ),
)

optimizer = optax.chain(
    optax.adam(learning_rate=cfg.learning_rate, eps=1e-5),
)

buffer = fbx.make_flat_buffer(
    max_length=cfg.buffer_size,
    min_length=cfg.batch_size,
    sample_batch_size=cfg.batch_size,
    add_sequences=False,
    add_batch_size=cfg.num_envs,
)

epsilon_schedule = optax.linear_schedule(
    cfg.start_e,
    cfg.end_e,
    int(cfg.exploration_fraction * total_timesteps),
    cfg.learning_starts,
)

key = jax.random.key(0)

agent = DQN(
    cfg=cfg,
    env=env,
    env_params=env_params,
    q_network=q_network,
    optimizer=optimizer,
    buffer=buffer,
    epsilon_schedule=epsilon_schedule,
)

logger = Logger([DashboardLogger(title="DQN Example", total_timesteps=total_timesteps)])
logger_state = logger.init(cfg)

key, state = agent.init(key)

for i in range(0, total_timesteps, num_train_steps):
    key, state, transitions = agent.train(key, state, num_steps=num_train_steps)

    training_statistics = Logger.get_episode_statistics(transitions, "training")
    data = {**training_statistics, **transitions.losses}
    logger_state = logger.log(logger_state, data, step=state.step.item())

    key, transitions = agent.evaluate(key, state, num_steps=num_eval_steps)
    evaluation_statistics = Logger.get_episode_statistics(transitions, "evaluation")
    logger_state = logger.log(
        logger_state, evaluation_statistics, step=state.step.item()
    )
    logger.emit(logger_state)
