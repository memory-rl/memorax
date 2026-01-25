# Quick Start

This guide demonstrates how to train a PPO agent with GRU memory on CartPole.

```python
from dataclasses import asdict

import flax.linen as nn
import jax
import optax

from memorax.algorithms import PPO, PPOConfig
from memorax.environments import environment
from memorax.loggers import DashboardLogger, Logger
from memorax.networks import MLP, FeatureExtractor, Network, RNN, heads

# Create environment
env, env_params = environment.make("gymnax::CartPole-v1")

# Configure PPO
cfg = PPOConfig(
    name="PPO-GRU",
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

# Build networks
d_model = 64

feature_extractor = FeatureExtractor(
    observation_extractor=MLP(
        features=(d_model,), kernel_init=nn.initializers.orthogonal(scale=1.414)
    ),
)
torso = RNN(cell=nn.GRUCell(features=d_model))
actor_network = Network(
    feature_extractor=feature_extractor,
    torso=torso,
    head=heads.Categorical(
        action_dim=env.action_space(env_params).n,
        kernel_init=nn.initializers.orthogonal(scale=0.01),
    ),
)
critic_network = Network(
    feature_extractor=feature_extractor,
    torso=torso,
    head=heads.VNetwork(kernel_init=nn.initializers.orthogonal(scale=1.0)),
)

# Create optimizer
optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adam(learning_rate=3e-4, eps=1e-5),
)

# Initialize agent and logger
agent = PPO(
    cfg=cfg,
    env=env,
    env_params=env_params,
    actor_network=actor_network,
    critic_network=critic_network,
    actor_optimizer=optimizer,
    critic_optimizer=optimizer,
)
logger = Logger([DashboardLogger(title="PPO-GRU CartPole", total_timesteps=500_000)])
logger_state = logger.init(cfg=asdict(cfg))

# Train
key = jax.random.key(0)
key, state = agent.init(key)

for i in range(0, 500_000, 10_000):
    key, state, transitions = agent.train(key, state, num_steps=10_000)
    training_statistics = Logger.get_episode_statistics(transitions, "training")
    logger_state = logger.log(logger_state, training_statistics, step=state.step.item())
    logger.emit(logger_state)

logger.finish(logger_state)
```

## Next Steps

- Learn about different {doc}`../guides/algorithms`
- Explore available {doc}`../guides/sequence_models`
- Build custom {doc}`../guides/networks`
