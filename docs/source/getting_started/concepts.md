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
import flax.linen as nn
from memorax.networks import SequenceModelWrapper, RNN, MLP, Memoroid, MambaCell

# GRU torso (recurrent models don't need wrapper)
torso = RNN(cell=nn.GRUCell(features=64))

# Mamba torso (MambaCell is used inside Memoroid)
torso = Memoroid(cell=MambaCell(features=64))

# MLP torso (non-recurrent models need SequenceModelWrapper)
torso = SequenceModelWrapper(MLP(features=(64,)))
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

## Buffers

Memorax provides episode-aware replay buffers for off-policy algorithms:

### Episode Buffer

Samples complete sequences while respecting episode boundaries:

```python
from memorax.buffers import make_episode_buffer

buffer = make_episode_buffer(
    max_length=100_000,
    min_length=1000,
    sample_batch_size=32,
    sample_sequence_length=16,
    add_batch_size=8,
)
```

### Prioritized Episode Buffer

Combines episode-aware sampling with Prioritized Experience Replay (PER):

```python
from memorax.buffers import make_prioritised_episode_buffer, compute_importance_weights

buffer = make_prioritised_episode_buffer(
    max_length=100_000,
    min_length=1000,
    sample_batch_size=32,
    sample_sequence_length=16,
    add_batch_size=8,
    priority_exponent=0.6,  # alpha: 0=uniform, 1=full prioritization
)

# Sampling returns indices and probabilities for importance weighting
sample = buffer.sample(state, key)
weights = compute_importance_weights(sample.probabilities, buffer_size, beta=0.4)

# Update priorities after computing TD-errors
state = buffer.set_priorities(state, sample.indices, jnp.abs(td_errors) + 1e-6)
```

Key features:
- Only samples from valid episode start positions
- Weights sampling proportionally to TD-error priorities
- Provides importance sampling weights to correct for non-uniform sampling
