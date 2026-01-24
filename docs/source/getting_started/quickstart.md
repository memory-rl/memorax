# Quick Start

This guide demonstrates how to train a PPO agent with LSTM memory on CartPole.

## Basic Training Loop

```python
import jax
import optax
from flax import nnx
from memorax.algorithms import PPO, PPOConfig
from memorax.environments import make
from memorax.networks import (
    MLP, RNN, FeatureExtractor, Network, SequenceModelWrapper, heads
)

# Create environment
env, env_params = make("gymnax::CartPole-v1")
action_dim = env.action_space(env_params).n

# Configure PPO
cfg = PPOConfig(
    name="PPO-LSTM",
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

# Build actor network
feature_extractor = FeatureExtractor(
    observation_extractor=MLP(features=(64,))
)
torso = SequenceModelWrapper(RNN(nnx.LSTMCell(64, 64, rngs=nnx.Rngs(0))))
actor = Network(feature_extractor, torso, heads.Categorical(action_dim))

# Build critic network (can share torso or use separate)
critic = Network(feature_extractor, torso, heads.VNetwork())

# Create optimizer
optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adam(3e-4)
)

# Initialize agent
agent = PPO(cfg, env, env_params, actor, critic, optimizer, optimizer)
key, state = agent.init(jax.random.key(0))

# Train
key, state, transitions = agent.train(key, state, num_steps=100_000)
```

## Evaluation

```python
# Evaluate the trained agent
key, eval_returns = agent.evaluate(key, state, num_episodes=10)
print(f"Mean evaluation return: {eval_returns.mean()}")
```

## Adding Logging

```python
from memorax.loggers import Logger, WandbLogger, ConsoleLogger

# Create logger
logger = Logger([WandbLogger(), ConsoleLogger()])
logger_state = logger.init(cfg=cfg)

# Training loop with logging
for step in range(0, 100_000, cfg.num_steps * cfg.num_envs):
    key, state, transitions = agent.train(key, state, num_steps=cfg.num_steps)

    # Log metrics
    logger_state = logger.log(logger_state, {
        "return": transitions.info["return"].mean(),
        "step": step,
    }, step=step)
    logger.emit(logger_state)

logger.finish(logger_state)
```

## Next Steps

- Learn about different {doc}`../guides/algorithms`
- Explore available {doc}`../guides/sequence_models`
- Build custom {doc}`../guides/networks`
