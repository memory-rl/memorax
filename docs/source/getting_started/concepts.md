# Core Concepts

This page explains the key concepts and design patterns in Memorax.

## Architecture Overview

Memorax follows a modular architecture where agents are composed of:

```
Agent = Algorithm + Network + Environment
Network = FeatureExtractor -> Torso -> Head
```

## Algorithms

Each algorithm in Memorax consists of three components:

- **Config**: A frozen dataclass containing hyperparameters
- **State**: A dataclass holding the training state (parameters, optimizer state, etc.)
- **Algorithm**: The main class implementing `init()`, `train()`, `warmup()`, and `evaluate()`

```python
from memorax.algorithms import PPO, PPOConfig, PPOState

# Configure
cfg = PPOConfig(num_envs=8, gamma=0.99)

# Create algorithm
agent = PPO(cfg, env, env_params, actor, critic, optimizer, optimizer)

# Initialize state
key, state = agent.init(key)

# Train
key, state, transitions = agent.train(key, state, num_steps=1000)
```

## Networks

Networks in Memorax are composed of three parts:

### Feature Extractor

Extracts features from observations, actions, rewards, and done flags:

```python
from memorax.networks import FeatureExtractor, MLP

feature_extractor = FeatureExtractor(
    observation_extractor=MLP(features=(64, 64)),
    action_extractor=None,  # Optional
    reward_extractor=None,  # Optional
    done_extractor=None,    # Optional
)
```

### Torso (Sequence Model)

Processes temporal sequences using RNNs, SSMs, or attention:

```python
from memorax.networks import SequenceModelWrapper, RNN
from memorax.networks.sequence_models import Mamba

# LSTM torso
torso = SequenceModelWrapper(RNN(nnx.LSTMCell(64, 64, rngs=nnx.Rngs(0))))

# Mamba torso
torso = SequenceModelWrapper(Mamba(hidden_dim=64))
```

### Head

Produces outputs for the RL objective:

```python
from memorax.networks import heads

# Discrete policy
head = heads.Categorical(action_dim=4)

# Continuous policy
head = heads.SquashedGaussian(action_dim=2)

# Value function
head = heads.VNetwork()

# Q-network
head = heads.DiscreteQNetwork(action_dim=4)
```

## JAX Patterns

Memorax leverages JAX for performance:

### Vectorized Environments

All training runs multiple environments in parallel:

```python
cfg = PPOConfig(num_envs=8)  # 8 parallel environments
```

### Random Keys

JAX uses explicit random state management:

```python
key = jax.random.key(0)
key, state = agent.init(key)
key, state, transitions = agent.train(key, state, num_steps=1000)
```

### JIT Compilation

Training loops are JIT-compiled for performance. The first call may be slow due to compilation.

## Transitions

Training produces `Transition` objects containing:

- `observation`: Environment observations
- `action`: Actions taken
- `reward`: Rewards received
- `done`: Episode termination flags
- `info`: Additional information (returns, etc.)
