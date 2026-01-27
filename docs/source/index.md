# Memorax

A unified JAX/Flax framework for memory-augmented reinforcement learning.

```{image} https://github.com/memory-rl/memorax/blob/main/images/memorax_logo.png?raw=true
:alt: Memorax Logo
:align: center
:height: 170px
```

Memorax provides modular, high-performance implementations of RL algorithms with support for advanced sequence models including RNNs, State Space Models, and Transformers.

---

## Features

**Algorithms**
: JAX implementations of [DQN](https://arxiv.org/abs/1312.5602), [PPO](https://arxiv.org/abs/1707.06347), [SAC](https://arxiv.org/abs/1801.01290), [PQN](https://arxiv.org/abs/2407.04811v2), [IPPO](https://arxiv.org/abs/2011.09533), [R2D2](https://openreview.net/forum?id=r1lyTjAqYX), and their memory-augmented variants with burn-in support.

**Sequence Models**
: LSTM/GRU (Flax), sLSTM/mLSTM, FFM/SHM, S5/LRU/Mamba/MinGRU, plus attention blocks (Self-Attention, Linear Attention). GPT-2/GTrXL/xLSTM-style architectures are composed from these blocks (see examples).

**Networks**
: MLP, CNN, and [ViT](https://arxiv.org/abs/2010.11929) encoders with [RoPE](https://arxiv.org/abs/2104.09864) and [ALiBi](https://arxiv.org/abs/2108.12409) positional embeddings, plus [Mixture of Experts](https://arxiv.org/abs/1701.06538) for horizontal scaling.

**Environments**
: Integration with [Gymnax](https://github.com/RobertTLange/gymnax), [PopJym](https://github.com/EdanToledo/popjym), [Navix](https://github.com/epignatelli/navix), [Craftax](https://github.com/MichaelTMatthews/Craftax), [Brax](https://github.com/google/brax), [MuJoCo](https://github.com/google-deepmind/mujoco_playground), [XMiniGrid](https://github.com/corl-team/xland-minigrid), and [JaxMARL](https://github.com/FLAIROx/JaxMARL).

**Logging**
: Built-in support for [Weights & Biases](https://wandb.ai), [TensorBoard](https://github.com/lanpa/tensorboardX), [Neptune](https://neptune.ai), and CLI dashboard.

---

## Installation

Install Memorax using pip:

```bash
pip install memorax
```

Or with CUDA support:

```bash
pip install memorax[cuda]
```

See the {doc}`getting_started/installation` guide for more options.

---

## Quick Example

```python
import jax
import optax
from memorax.algorithms import PPO, PPOConfig
from memorax.environments import environment
from memorax.networks import MLP, FeatureExtractor, Network, heads

env, env_params = environment.make("gymnax::CartPole-v1")

cfg = PPOConfig(
    name="PPO",
    num_envs=8,
    num_steps=128,
    gamma=0.99,
    gae_lambda=0.95,
)

feature_extractor = FeatureExtractor(observation_extractor=MLP(features=(64,)))
actor = Network(feature_extractor, heads.Categorical(env.action_space(env_params).n))
critic = Network(feature_extractor, heads.VNetwork())
optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(3e-4))

agent = PPO(cfg, env, env_params, actor, critic, optimizer, optimizer)
key, state = agent.init(jax.random.key(0))
key, state, transitions = agent.train(key, state, num_steps=10_000)
```

See the {doc}`getting_started/quickstart` for more examples.

---

## Citation

If you use Memorax in your research, please cite:

```bibtex
@software{memoryrl2025github,
  title   = {Memory-RL: A Unified Framework for Memory-Augmented Reinforcement Learning},
  author  = {Noah Farr},
  year    = {2025},
  url     = {https://github.com/memory-rl/memorax}
}
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
