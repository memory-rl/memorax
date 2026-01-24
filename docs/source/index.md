# Memorax

A unified JAX/Flax framework for memory-augmented reinforcement learning.

Memorax provides modular, high-performance implementations of RL algorithms with support for advanced sequence models including RNNs, State Space Models, and Transformers.

## Features

- **Algorithms**: PPO, DQN, SAC, PQN with full JAX vectorization
- **Sequence Models**: LSTM, GRU, Mamba, S5, LRU, Linear Attention, and more
- **Environments**: Integration with Gymnax, Brax, POPGym, Craftax, and others
- **Logging**: Weights & Biases, TensorBoard, Neptune, and console logging

## Installation

```bash
pip install memorax
```

## Quick Start

```python
import jax
import optax
from flax import nnx
from memorax.algorithms import PPO, PPOConfig
from memorax.environments import environment
from memorax.networks import (
    MLP, RNN, FeatureExtractor, Network, SequenceModelWrapper, heads
)

# Create environment
env, env_params = environment.make("gymnax::CartPole-v1")

# Configure PPO
cfg = PPOConfig(
    name="PPO-LSTM",
    num_envs=8,
    num_steps=128,
    gamma=0.99,
)

# Build networks with LSTM memory
feature_extractor = FeatureExtractor(
    observation_extractor=MLP(features=(64,))
)
torso = SequenceModelWrapper(RNN(nnx.LSTMCell(64, 64, rngs=nnx.Rngs(0))))
actor = Network(feature_extractor, torso, heads.Categorical(env.action_space(env_params).n))
critic = Network(feature_extractor, torso, heads.VNetwork())

# Create optimizer
optimizer = optax.adam(3e-4)

# Initialize and train
agent = PPO(cfg, env, env_params, actor, critic, optimizer, optimizer)
key, state = agent.init(jax.random.key(0))
key, state, transitions = agent.train(key, state, num_steps=100_000)
```

```{toctree}
:maxdepth: 2
:caption: Getting Started
:hidden:

getting_started/installation
getting_started/quickstart
getting_started/concepts
```

```{toctree}
:maxdepth: 2
:caption: User Guides
:hidden:

guides/algorithms
guides/networks
guides/sequence_models
```

```{toctree}
:maxdepth: 3
:caption: API Reference
:hidden:

api/index
```
