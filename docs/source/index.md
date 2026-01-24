# Memory-Augmented Reinforcement Learning in JAX

A unified JAX/Flax framework for memory-augmented reinforcement learning.

Memorax provides modular, high-performance implementations of RL algorithms with support for advanced sequence models including RNNs, State Space Models, and Transformers.

## Features

- 🧠 **Algorithms**: `PPO`, `DQN`, `SAC`, `PQN` with full JAX vectorization
- 🔁 **Sequence Models**: `LSTM`, `GRU`, `Mamba`, `S5`, `LRU`, `Linear Attention`, and more
- 🌍 **Environments**: Integration with Gymnax, Brax, POPGym, Craftax, and others
- 📊 **Logging**: Weights & Biases, TensorBoard, Neptune, and console logging

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
