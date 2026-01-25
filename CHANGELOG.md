# Changelog

All notable changes to Memorax will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]


## [1.0.0] - 2026-01-25

### Added

#### Algorithms
- DQN with Double DQN and Dueling DQN variants
- PPO for both discrete and continuous action spaces
- SAC (Soft Actor-Critic) for continuous control
- PQN (Policy Q-Network) for on-policy Q-learning

#### Sequence Models
- Classical RNNs: LSTM, GRU (via Flax wrappers)
- Advanced RNNs: sLSTM, mLSTM, xLSTM
- State Space Models: LRU, S5, Mamba, MinGRU
- Memory architectures: FFM, SHM (Stable Hadamard Memory), Memoroid
- Attention: Linear Transformer, Self-Attention with segment-level recurrence
- GTrXL (Gated Transformer-XL) support

#### Network Components
- Feature extractors: MLP, CNN, ViT
- Positional embeddings: RoPE, ALiBi, Learnable
- Normalization: PreNorm, PostNorm, LayerNorm
- Residual connections: Residual, GatedResidual
- Feed-forward networks with configurable expansion
- Mixture of Experts (MoE) routing for horizontal scaling

#### Environments
- Gymnax integration (classic control, Atari-style)
- Brax integration (continuous control, physics)
- PopGym/PopJym integration (memory benchmarks)
- PopGym Arcade integration
- Craftax integration (multi-task learning)
- Navix integration (maze navigation)
- MuJoCo Playground integration
- XMiniGrid integration (grid worlds)
- gxm integration (custom environments)

#### Logging
- Console logger
- CLI Dashboard logger
- Weights & Biases integration
- TensorBoard integration
- Neptune integration
- File logger

#### Infrastructure
- JAX-native implementation with full JIT/vmap/scan support
- Flashbax-based replay buffers
- Pre-commit hooks for code quality
- Comprehensive test suite (104 tests)
- 20 example scripts

