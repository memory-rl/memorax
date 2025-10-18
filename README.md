# Memory-RL Framework

A unified reinforcement learning framework featuring memory-augmented algorithms and POMDP environment implementations. This repository provides modular components for building, configuring, and running a variety of RL algorithms on classic and memory-intensive environments.

## Features

* **Memory-RL**: JAX implementations of [DQN](https://arxiv.org/abs/1312.5602), [PPO](https://arxiv.org/abs/1707.06347) (Discrete & Continuous), [SAC](https://arxiv.org/abs/1801.01290) (Discrete and Continuous), [PQN](https://arxiv.org/abs/2407.04811v2#S4), and their memory-augmented variants.
* **Recurrent Cells**: Support for multiple RNN cells and Memory Architectures, including [LSTM](https://ieeexplore.ieee.org/abstract/document/6795963), [GRU](https://arxiv.org/abs/1412.3555), [GPT2](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf), [GTrXL](https://arxiv.org/abs/1910.06764), [FFM](https://arxiv.org/abs/2310.04128), [xLSTM](https://arxiv.org/abs/2405.04517), [SHM](https://arxiv.org/abs/2410.10132) and [S5](https://arxiv.org/abs/2303.03982).
* **Environments**: Support for [Gymnax](https://github.com/RobertTLange/gymnax), [PopJym](https://github.com/EdanToledo/popjym), [PopGym Arcade](https://github.com/bolt-research/popgym-arcade), [Navix](https://github.com/epignatelli/navix?tab=readme-ov-file), [Craftax](https://github.com/MichaelTMatthews/Craftax), [Brax](https://github.com/google/brax), [MuJoCo](https://github.com/google-deepmind/mujoco_playground) and [gxm](https://github.com/huterguier/gxm).
* **Logging & Sweeps**: Support for a CLI Dashboard, [Weights & Biases](https://wandb.ai), [TensorboardX](https://github.com/lanpa/tensorboardX) and more.
* **Easy to Extend**: Clear directory structure for adding new networks, algorithms, or environments.

## Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/memory-rl/memory-rl.git
   cd memory-rl
   ```

2. **Install Python dependencies**:

   ```bash
   uv sync
   ```
Optionally you can add support for CUDA with
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
uv run examples/dqn_gymnax.py
```

## Project structure
```
memory-rl/
├─ examples/          # Small runnable scripts (e.g., DQN CartPole)
├─ memory_rl/
   ├─ algorithms/     # DQN, PPO, SAC, PQN, ...
   ├─ networks/       # MLP, CNN, RNN, heads, ...
   ├─ environments/   # Gymnax / PopGym / Brax / ...
   ├─ buffers/        # Custom flashbax buffers
   ├─ loggers/        # CLI, WandB, TensorBoardX integrations
   └─ utils/
```

## License

...

## Citation

If you use Memory-RL for your work, please cite:

@software{memoryrl2025github,
  title   = {Memory-RL: A Unified Framework for Memory-Augmented Reinforcement Learning},
  author  = {Noah Farr},
  year    = {2025},
  url     = {https://github.com/memory-rl/memory-rl}
}
