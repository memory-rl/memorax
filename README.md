# 🧠 Memorax

A unified reinforcement learning framework featuring memory-augmented algorithms and POMDP environment implementations. This repository provides modular components for building, configuring, and running a variety of RL algorithms on classic and memory-intensive environments.

<div align="center">
    <img src="https://github.com/memory-rl/memorax/blob/main/images/memorax_logo.png" height="170">
</div>

## ✨ Features

* 🤖 **Memory-RL**: JAX implementations of [DQN](https://arxiv.org/abs/1312.5602), [PPO](https://arxiv.org/abs/1707.06347) (Discrete & Continuous), [SAC](https://arxiv.org/abs/1801.01290) (Discrete & Continuous), [PQN](https://arxiv.org/abs/2407.04811v2#S4), [IPPO](https://arxiv.org/abs/2011.09533), [R2D2](https://openreview.net/forum?id=r1lyTjAqYX), and their memory-augmented variants with burn-in support for recurrent networks.
* 📦 **Pure JAX Episode Buffer**: A fully JAX-native episode buffer implementation enabling efficient storage and sampling of complete episodes for recurrent training, with support for [Prioritized Experience Replay](https://arxiv.org/abs/1511.05952).
* 🔁 **Recurrent Cells**: Support for multiple RNN cells and Memory Architectures, including [LSTM](https://ieeexplore.ieee.org/abstract/document/6795963), [GRU](https://arxiv.org/abs/1412.3555), [GPT2](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf), [GTrXL](https://arxiv.org/abs/1910.06764), [FFM](https://arxiv.org/abs/2310.04128), [xLSTM](https://arxiv.org/abs/2405.04517), [SHM](https://arxiv.org/abs/2410.10132), [S5](https://arxiv.org/abs/2303.03982), [LRU](https://arxiv.org/abs/2303.06349), [RetNet](https://arxiv.org/abs/2307.08621), [Mamba](https://arxiv.org/abs/2312.00752), [MinGRU](https://arxiv.org/abs/2410.01201), [Linear Transformer](https://arxiv.org/abs/2006.16236).
* 🧬 **Networks**: MLP, CNN, and [ViT](https://arxiv.org/abs/2010.11929) encoders with support for [RoPE](https://arxiv.org/abs/2104.09864) and [ALiBi](https://arxiv.org/abs/2108.12409) positional embeddings, and [Mixture of Experts (MoE)](https://arxiv.org/abs/1701.06538) for horizontal scaling.
* 🎮 **Environments**: Support for [Gymnax](https://github.com/RobertTLange/gymnax), [PopJym](https://github.com/EdanToledo/popjym), [PopGym Arcade](https://github.com/bolt-research/popgym-arcade), [Navix](https://github.com/epignatelli/navix), [Craftax](https://github.com/MichaelTMatthews/Craftax), [Brax](https://github.com/google/brax), [MuJoCo](https://github.com/google-deepmind/mujoco_playground), [gxm](https://github.com/huterguier/gxm), [XMiniGrid](https://github.com/corl-team/xland-minigrid), and [JaxMARL](https://github.com/FLAIROx/JaxMARL).
* 📊 **Logging & Sweeps**: Support for a CLI Dashboard, [Weights & Biases](https://wandb.ai), [TensorboardX](https://github.com/lanpa/tensorboardX), and [Neptune](https://neptune.ai).
* 🔧 **Easy to Extend**: Clear directory structure for adding new networks, algorithms, or environments.

## 📥 Installation

1. **Clone the repository**:

```bash
git clone https://github.com/memorax/memorax.git
cd memorax
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

## 🚀 Quick Start

Run a default DQN experiment on CartPole:

```bash
uv run examples/dqn_gymnax.py
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
