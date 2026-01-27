# 🧠 Memorax

A unified reinforcement learning framework featuring memory-augmented algorithms and POMDP environment implementations. This repository provides modular components for building, configuring, and running a variety of RL algorithms on classic and memory-intensive environments.

<div align="center">
    <img src="https://github.com/memory-rl/memorax/blob/main/images/memorax_logo.png" height="170">
</div>

## ✨ Features

* 🤖 **Memory-RL**: JAX implementations of [DQN](https://arxiv.org/abs/1312.5602), [PPO](https://arxiv.org/abs/1707.06347) (Discrete & Continuous), [SAC](https://arxiv.org/abs/1801.01290) (Discrete & Continuous), [PQN](https://arxiv.org/abs/2407.04811v2#S4), [IPPO](https://arxiv.org/abs/2011.09533), [R2D2](https://openreview.net/forum?id=r1lyTjAqYX), and their memory-augmented variants with burn-in support for recurrent networks.
* 📦 **Pure JAX Episode Buffer**: A fully JAX-native episode buffer implementation enabling efficient storage and sampling of complete episodes for recurrent training, with support for [Prioritized Experience Replay](https://arxiv.org/abs/1511.05952).
* 🔁 **Sequence Models**: LSTM/GRU (via Flax), sLSTM/mLSTM, FFM/SHM, S5/LRU/Mamba/MinGRU, plus Self-Attention and Linear Attention blocks. GPT-2/GTrXL/xLSTM-style architectures are composed from these primitives (see `examples/architectures`).
* 🧬 **Networks**: MLP, CNN, and [ViT](https://arxiv.org/abs/2010.11929) encoders with support for [RoPE](https://arxiv.org/abs/2104.09864) and [ALiBi](https://arxiv.org/abs/2108.12409) positional embeddings, and [Mixture of Experts (MoE)](https://arxiv.org/abs/1701.06538) for horizontal scaling.
* 🎮 **Environments**: Support for [Gymnax](https://github.com/RobertTLange/gymnax), [PopJym](https://github.com/EdanToledo/popjym), [PopGym Arcade](https://github.com/bolt-research/popgym-arcade), [Navix](https://github.com/epignatelli/navix), [Craftax](https://github.com/MichaelTMatthews/Craftax), [Brax](https://github.com/google/brax), [MuJoCo](https://github.com/google-deepmind/mujoco_playground), [gxm](https://github.com/huterguier/gxm), [XMiniGrid](https://github.com/corl-team/xland-minigrid), and [JaxMARL](https://github.com/FLAIROx/JaxMARL).
* 📊 **Logging & Sweeps**: Support for a CLI Dashboard, [Weights & Biases](https://wandb.ai), [TensorboardX](https://github.com/lanpa/tensorboardX), and [Neptune](https://neptune.ai).
* 🔧 **Easy to Extend**: Clear directory structure for adding new networks, algorithms, or environments.

## 📥 Installation

Install Memorax using pip:

```bash
pip install memorax
```

Or using uv:

```bash
uv add memorax
```

Optionally you can add support for CUDA with:

```bash
pip install memorax[cuda]
```

**Optional**: Set up Weights & Biases for logging by logging in:

```bash
wandb login
```

## 🚀 Quick Start

Run a default DQN experiment on CartPole:

```bash
uv run examples/dqn_cartpole.py
```

## 💻 Usage

```python
import jax
import optax
from memorax.algorithms import PPO, PPOConfig
from memorax.environments import environment
from memorax.networks import (
    MLP, FFN, ALiBi, FeatureExtractor, GatedResidual, Network,
    PreNorm, SegmentRecurrence, SelfAttention, Stack, heads,
)

env, env_params = environment.make("gymnax::CartPole-v1")

cfg = PPOConfig(
    name="PPO-GTrXL",
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

features, num_heads, num_layers = 64, 4, 2
feature_extractor = FeatureExtractor(observation_extractor=MLP(features=(features,)))
attention = GatedResidual(PreNorm(SegmentRecurrence(
    SelfAttention(features, num_heads, context_length=128, positional_embedding=ALiBi(num_heads)),
    memory_length=64, features=features,
)))
ffn = GatedResidual(PreNorm(FFN(features=features, expansion_factor=4)))
torso = Stack(blocks=(attention, ffn) * num_layers)

actor_network = Network(feature_extractor, torso, heads.Categorical(env.action_space(env_params).n))
critic_network = Network(feature_extractor, torso, heads.VNetwork())
optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(3e-4))

agent = PPO(cfg, env, env_params, actor_network, critic_network, optimizer, optimizer)
key, state = agent.init(jax.random.key(0))
key, state, transitions = agent.train(key, state, num_steps=10_000)
```

## 📂 Project Structure
```
memorax/
├─ examples/          # Small runnable scripts (e.g., DQN CartPole)
├─ memorax/
   ├─ algorithms/     # DQN, PPO, SAC, PQN, ...
   ├─ networks/       # MLP, CNN, ViT, RNN, heads, ...
   ├─ environments/   # Gymnax / PopGym / Brax / ...
   ├─ buffers/        # Custom flashbax buffers
   ├─ loggers/        # CLI, WandB, TensorBoardX integrations
   └─ utils/
```

## 🧩 JAX POMDP Ecosystem

Memorax is designed to work alongside a growing suite of JAX-native tools focused on partial observability and memory. These projects provide the foundational architectures and benchmarks for modern memory-augmented RL:

### 🧠 Architectures & Infrastructure
* **[Memax](https://github.com/smorad/memax):** A library for efficient sequence and recurrent modeling in JAX. It provides unified interfaces for fast recurrent state resets and associative scans, serving as a powerful primitive for building memory architectures.
* **[Flashbax](https://github.com/instadeepai/flashbax):** The library powering Memorax's buffer system. It provides high-performance, JAX-native experience replay buffers optimized for sequence storage and prioritized sampling.
* **[Gymnax](https://github.com/RobertTLange/gymnax):** The standard for JAX-native RL environments. Memorax provides seamless wrappers to run recurrent agents on these vectorized tasks.

### 🎮 POMDP Benchmarks & Environments
* **[PopGym Arcade](https://github.com/bolt-lab/popgym-arcade):** A JAX-native suite of "pixel-perfect" POMDP environments. It features Atari-style games specifically designed to test long-term memory with hardware-accelerated rendering.
* **[PopJym](https://github.com/bolt-lab/popjym):** A fast, JAX-native implementation of the POPGym benchmark suite, providing a variety of classic POMDP tasks optimized for massive vectorization.
* **[Navix](https://github.com/pignatelli/navix):** Accelerated MiniGrid-style environments. These are excellent for testing spatial reasoning and navigation in partially observable grid worlds.
* **[XLand-MiniGrid](https://github.com/corl-team/xland-minigrid):** A high-throughput meta-RL environment suite that provides massive task diversity for testing agent generalization in POMDPs.

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## 📚 Citation

If you use Memory-RL for your work, please cite:
```
@software{memoryrl2025github,
  title   = {Memory-RL: A Unified Framework for Memory-Augmented Reinforcement Learning},
  author  = {Noah Farr},
  year    = {2025},
  url     = {https://github.com/memory-rl/memorax}
}
```

## 🙏 Acknowledgments

Special thanks to [@huterguier](https://github.com/huterguier) for the valuable discussions and advice on the API design.
