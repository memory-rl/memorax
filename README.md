# Memory-RL Framework

A unified reinforcement learning framework featuring memory-augmented algorithms and POMDP environment implementations. This repository provides modular components for building, configuring, and running a variety of RL algorithms on classic and memory-intensive environments.

## Features

* **Memory-RL**: JAX implementations of DQN, DRQN, PPO, SAC, SAC-D, PPO, and their memory-augmented variants.
* **Recurrent Cells**: Support for multiple RNN cells, including LSTM, GRU, GPT2, FFM.
* **Environments**: Support for Gymnax, bsuite, PopGym, and custom T-maze tasks.
* **Configurable**: YAML-driven configuration for algorithms, environments, hyperparameters, logging, and sweep setups.
* **Logging & Sweeps**: Integrated with Hydra and Weights & Biases for experiment management.
* **Easy to Extend**: Clear directory structure for adding new networks, algorithms, or environments.

## Installation

1. **Clone the repository**:

   ```bash
   git clone --recurse-submodules https://github.com/memory-rl/memory-rl.git
   cd memory-rl
   ```

2. **Install Python dependencies**:

   ```bash
   uv sync
   ```
3. **Install jax-cuda**:
   
   ```bash
   uv sync --extra cuda
   ```

3. **Optional**: Set up Weights & Biases for logging by logging in:

   ```bash
   wandb login
   ```

## Quick Start

Run a default DQN experiment on CartPole:

```bash
uv run main algorithm=dqn environment=cartpole
```

Launch RPPO run on repeat-recall with wandb logging and set seed:

```bash
uv run main environment repeat_first algorithm rppo logger=wandb seed=42
```

## Configuration

All settings are managed via YAML files in `memory_rl/conf`. Use Hydra overrides to customize.

Key config groups:

* **algorithm**: selects and parametrizes RL algorithms.
* **environment**: specifies task and environment settings.
* **hyperparameter**: workout preset hyperparameters for common tasks.
* **logger**: toggle between `default`, `console` and `wandb` logging.

## Outputs & Logging

Experiment logs and checkpoints are saved under `outputs/YYYY-MM-DD/HH-MM-SS`:

* `main.log`: training and evaluation logs.
* Model checkpoints in the working directory.
