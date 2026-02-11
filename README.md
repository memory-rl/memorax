<h1 align="center">
  <a href="https://github.com/memory-rl/memorax/blob/main/images/memorax_logo.png"><img src="https://github.com/memory-rl/memorax/blob/main/images/memorax_logo.png?raw=true" /></a><br>
  <b>Memory-Augmented Reinforcement Learning in JAX 🧠</b><br>
</h1>

<p align="center">
  <a href="https://pypi.org/project/memorax/"><img src="https://img.shields.io/pypi/v/memorax.svg" /></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.11+-blue.svg" /></a>
  <a href="https://opensource.org/licenses/Apache-2.0"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" /></a>
  <a href="https://memorax.readthedocs.io/"><img src="https://img.shields.io/readthedocs/memorax" /></a>
</p>

Most JAX RL libraries treat memory as an afterthought, bolting an LSTM onto an existing agent and calling it done. `Memorax` makes memory a first-class citizen. It provides a composable set of sequence model primitives (attention, SSMs, linear RNNs, and more) that snap together into full architectures like `GTrXL` or `xLSTM`, paired with algorithms and replay buffers designed from the ground up for recurrent training. Whether you're benchmarking a new memory architecture on POMDPs or scaling recurrent agents across environments, `Memorax` gives you the building blocks to do it entirely in JAX.

## ✨ Features

| | Details |
|---|---|
| 🤖 **Algorithms** | [DQN](https://arxiv.org/abs/1312.5602), [PPO](https://arxiv.org/abs/1707.06347), [SAC](https://arxiv.org/abs/1801.01290), [PQN](https://arxiv.org/abs/2407.04811v2#S4), [IPPO](https://arxiv.org/abs/2011.09533), [IRPPO](https://arxiv.org/abs/1810.04650), [R2D2](https://openreview.net/forum?id=r1lyTjAqYX) + memory-augmented variants with burn-in. [ICM](https://arxiv.org/abs/1705.05363) and [RND](https://arxiv.org/abs/1810.12894) for intrinsic motivation |
| 📦 **Buffers** | Pure JAX episode replay with prioritized sampling via [Flashbax](https://github.com/instadeepai/flashbax) |
| 🔁 **Sequence Models** | [LSTM](https://doi.org/10.1162/neco.1997.9.8.1735), [GRU](https://arxiv.org/abs/1406.1078), [xLSTM](https://arxiv.org/abs/2405.04517), [FFM](https://arxiv.org/abs/2310.04128), [SHM](https://arxiv.org/abs/2410.10132), [S5](https://arxiv.org/abs/2208.04933), [LRU](https://arxiv.org/abs/2303.06349), [Mamba](https://arxiv.org/abs/2312.00752), [MinGRU](https://arxiv.org/abs/2410.01201), [Self-Attention](https://arxiv.org/abs/1706.03762), [Linear Attention](https://arxiv.org/abs/2006.16236). Compose into `GTrXL`, `GPT-2`, and more. Support for [RTRL](https://doi.org/10.1162/neco.1989.1.2.270) |
| 🧬 **Networks** | [MLP](https://en.wikipedia.org/wiki/Multilayer_perceptron), [CNN](https://en.wikipedia.org/wiki/Convolutional_neural_network), and [ViT](https://arxiv.org/abs/2010.11929) encoders. [RoPE](https://arxiv.org/abs/2104.09864) and [ALiBi](https://arxiv.org/abs/2108.12409) positional embeddings. [MoE](https://arxiv.org/abs/1701.06538) for horizontal scaling. [RL²](https://arxiv.org/abs/1611.02779) wrapper for meta-RL. Composable `feature extractor` → `torso` → `head` pipeline |
| 🎮 **Environments** | [Gymnax](https://github.com/RobertTLange/gymnax), [PopJym](https://github.com/EdanToledo/popjym), [PopGym Arcade](https://github.com/bolt-research/popgym-arcade), [Navix](https://github.com/epignatelli/navix), [Craftax](https://github.com/MichaelTMatthews/Craftax), [Brax](https://github.com/google/brax), [MuJoCo](https://github.com/google-deepmind/mujoco_playground), [gxm](https://github.com/huterguier/gxm), [XMiniGrid](https://github.com/corl-team/xland-minigrid), [JaxMARL](https://github.com/FLAIROx/JaxMARL) |
| 📊 **Logging** | CLI Dashboard, [W&B](https://wandb.ai), [TensorboardX](https://github.com/lanpa/tensorboardX), [Neptune](https://neptune.ai) |

## 📥 Installation

Install `Memorax` using pip:

```bash
pip install memorax
```

Optionally you can add support for `CUDA` with:

```bash
pip install memorax[cuda]
```

**Optional**: Set up Weights & Biases for logging by logging in:

```bash
wandb login
```

## 🚀 Quick Start

Train a DQN agent on CartPole in under 30 lines:

```python
import flax.linen as nn
import jax
import optax
from flashbax import make_item_buffer
from memorax.algorithms import DQN, DQNConfig
from memorax.environments import environment
from memorax.networks import FeatureExtractor, Network, heads

env, env_params = environment.make("gymnax::CartPole-v1")

cfg = DQNConfig(
    name="dqn", num_envs=10, num_eval_envs=10, buffer_size=10_000,
    gamma=0.99, tau=1.0, target_network_frequency=500, batch_size=64,
    start_e=1.0, end_e=0.05, exploration_fraction=0.5, train_frequency=10,
)

q_network = Network(
    feature_extractor=FeatureExtractor(observation_extractor=nn.Sequential((nn.Dense(120), nn.relu, nn.Dense(84), nn.relu))),
    head=heads.DiscreteQNetwork(action_dim=env.action_space(env_params).n),
)

optimizer = optax.adam(3e-4)
buffer = make_item_buffer(max_length=cfg.buffer_size, min_length=cfg.batch_size,
                          sample_batch_size=cfg.batch_size, add_sequences=True, add_batches=True)
epsilon = optax.linear_schedule(cfg.start_e, cfg.end_e, 250_000, 10_000)

agent = DQN(cfg, env, env_params, q_network, optimizer, buffer, epsilon)
key, state = agent.init(jax.random.key(0))
key, state, transitions = agent.train(key, state, num_steps=500_000)
```

See `examples/` for complete scripts with logging and evaluation.

## 💡 Advanced Usage

`Memorax`'s real power is in its composable network primitives. Here's a `PPO` agent with a `GTrXL`-style architecture, built by snapping together modular blocks:

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

See `examples/architectures` for more architecture compositions including `xLSTM` and `GPT-2` style networks.

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

`Memorax` is designed to work alongside a growing suite of JAX-native tools focused on partial observability and memory. These projects provide the foundational architectures and benchmarks for modern memory-augmented RL:

### 🧠 Architectures & Infrastructure
* **[Memax](https://github.com/smorad/memax):** A library for efficient sequence and recurrent modeling in JAX. It provides unified interfaces for fast recurrent state resets and associative scans, serving as a powerful primitive for building memory architectures.
* **[Flashbax](https://github.com/instadeepai/flashbax):** The library powering `Memorax`'s buffer system. It provides high-performance, JAX-native experience replay buffers optimized for sequence storage and prioritized sampling.
* **[Gymnax](https://github.com/RobertTLange/gymnax):** The standard for JAX-native RL environments. `Memorax` provides seamless wrappers to run recurrent agents on these vectorized tasks.

### 🎮 POMDP Benchmarks & Environments
* **[PopGym Arcade](https://github.com/bolt-lab/popgym-arcade):** A JAX-native suite of "pixel-perfect" POMDP environments. It features Atari-style games specifically designed to test long-term memory with hardware-accelerated rendering.
* **[PopJym](https://github.com/bolt-lab/popjym):** A fast, JAX-native implementation of the POPGym benchmark suite, providing a variety of classic POMDP tasks optimized for massive vectorization.
* **[Navix](https://github.com/pignatelli/navix):** Accelerated MiniGrid-style environments. These are excellent for testing spatial reasoning and navigation in partially observable grid worlds.
* **[XLand-MiniGrid](https://github.com/corl-team/xland-minigrid):** A high-throughput meta-RL environment suite that provides massive task diversity for testing agent generalization in POMDPs.

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## 📚 Citation

If you use `Memorax` for your work, please cite:
```
@software{memorax2025github,
  title   = {Memorax: A Unified Framework for Memory-Augmented Reinforcement Learning},
  author  = {Noah Farr},
  year    = {2025},
  url     = {https://github.com/memory-rl/memorax}
}
```

## 🙏 Acknowledgments

Special thanks to [@huterguier](https://github.com/huterguier) for the valuable discussions and advice on the API design.
