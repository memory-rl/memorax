```{image} _static/memorax_logo.png
:alt: Memorax Logo
:width: 450px
:align: center
```

Most JAX RL libraries treat memory as an afterthought, bolting an LSTM onto an existing agent and calling it done. Memorax makes memory a first-class citizen. It provides a composable set of sequence model primitives (attention, SSMs, linear RNNs, and more) that snap together into full architectures like GTrXL or xLSTM, paired with algorithms and replay buffers designed from the ground up for recurrent training.

## Features

| | Details |
|---|---|
| **Algorithms** | [DQN](https://arxiv.org/abs/1312.5602), [PPO](https://arxiv.org/abs/1707.06347), [SAC](https://arxiv.org/abs/1801.01290), [PQN](https://arxiv.org/abs/2407.04811v2), [MAPPO](https://arxiv.org/abs/2103.01955), [R2D2](https://openreview.net/forum?id=r1lyTjAqYX), [AC(λ)](https://arxiv.org/abs/2410.14606), [GradientPPO](https://arxiv.org/abs/2507.09087) |
| **Sequence Models** | [LSTM](https://doi.org/10.1162/neco.1997.9.8.1735), [GRU](https://arxiv.org/abs/1406.1078), [xLSTM](https://arxiv.org/abs/2405.04517), [FFM](https://arxiv.org/abs/2310.04128), [SHM](https://arxiv.org/abs/2410.10132), [S5](https://arxiv.org/abs/2208.04933), [LRU](https://arxiv.org/abs/2303.06349), [Mamba](https://arxiv.org/abs/2312.00752), [MinGRU](https://arxiv.org/abs/2410.01201), [RTU](https://arxiv.org/abs/2409.01449), [Self-Attention](https://arxiv.org/abs/1706.03762), [Linear Attention](https://arxiv.org/abs/2006.16236). Compose into GTrXL, GPT-2, and more. Support for [RTRL](https://doi.org/10.1162/neco.1989.1.2.270) |
| **Networks** | [ViT](https://arxiv.org/abs/2010.11929) encoder. [RoPE](https://arxiv.org/abs/2104.09864) and [ALiBi](https://arxiv.org/abs/2108.12409) positional embeddings. [MoE](https://arxiv.org/abs/1701.06538) for horizontal scaling. [RL²](https://arxiv.org/abs/1611.02779) wrapper for meta-RL. [GVF/Horde](https://www.cs.swarthmore.edu/~meeden/DevelopmentalRobotics/horde1.pdf) heads. [C51](https://arxiv.org/abs/1707.06887) and [HL-Gauss](https://arxiv.org/abs/2403.03950) distributional value heads. Composable `feature extractor` → `torso` → `head` pipeline |
| **Environments** | [Gymnax](https://github.com/RobertTLange/gymnax), [PopJym](https://github.com/EdanToledo/popjym), [PopGym Arcade](https://github.com/bolt-research/popgym-arcade), [Navix](https://github.com/epignatelli/navix), [Craftax](https://github.com/MichaelTMatthews/Craftax), [Brax](https://github.com/google/brax), [MuJoCo](https://github.com/google-deepmind/mujoco_playground), [gxm](https://github.com/huterguier/gxm), [Grimax](https://github.com/noahfarr/grimax), [POBAX](https://github.com/taodav/pobax), [XMiniGrid](https://github.com/corl-team/xland-minigrid), [JaxMARL](https://github.com/FLAIROx/JaxMARL) |
| **Buffers** | Pure JAX episode replay with prioritized sampling via [Flashbax](https://github.com/instadeepai/flashbax) |
| **Logging** | CLI Dashboard, File, [W&B](https://wandb.ai), [TensorboardX](https://github.com/lanpa/tensorboardX), [Neptune](https://neptune.ai) |

## Installation

```bash
pip install memorax
```

With CUDA support:

```bash
pip install "memorax[cuda]"
```

See the {doc}`getting_started/installation` guide for more options.

## Quick Start

```python
import flax.linen as nn
import jax
import optax
from memorax.algorithms import PPO, PPOConfig
from memorax.environments import make
from memorax.networks import (
    FFN, ALiBi, FeatureExtractor, GatedResidual, Network,
    PreNorm, SegmentRecurrence, SelfAttention, Stack, heads,
)

env, env_params = make("gymnax::CartPole-v1")

config = PPOConfig(
    num_envs=8,
    num_steps=128,
    gae_lambda=0.95,
    num_minibatches=4,
    update_epochs=4,
    normalize_advantage=True,
    clip_coefficient=0.2,
    clip_value_loss=True,
    entropy_coefficient=0.01,
)

features, num_heads, num_layers = 64, 4, 2
feature_extractor = FeatureExtractor(observation_extractor=nn.Sequential((nn.Dense(features), nn.relu)))
attention = GatedResidual(PreNorm(SegmentRecurrence(
    SelfAttention(features, num_heads, context_length=128, positional_embedding=ALiBi(num_heads)),
    memory_length=64, features=features,
)))
ffn = GatedResidual(PreNorm(FFN(features=features, expansion_factor=4)))
torso = Stack(blocks=(attention, ffn) * num_layers)

actor = Network(feature_extractor, torso, heads.Categorical(env.action_space(env_params).n))
critic = Network(feature_extractor, torso, heads.VNetwork())
optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(3e-4))

agent = PPO(config, env, env_params, actor, critic, optimizer, optimizer)
key, state = agent.init(jax.random.key(0))
key, state = agent.train(key, state, num_steps=10_000)
```

See the {doc}`getting_started/quickstart` for a complete walkthrough.

## Citation

```bibtex
@software{memorax2025github,
  title   = {Memorax: A Unified Framework for Memory-Augmented Reinforcement Learning},
  author  = {Noah Farr},
  year    = {2025},
  url     = {https://github.com/noahfarr/memorax}
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
