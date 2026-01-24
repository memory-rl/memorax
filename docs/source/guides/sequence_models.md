# Sequence Models

This guide covers the available sequence models for memory-augmented RL.

## Overview

Memorax supports three families of sequence models:

| Family | Models | Strengths |
|--------|--------|-----------|
| RNNs | LSTM, GRU, sLSTM, mLSTM, xLSTM | Simple, well-understood |
| State Space Models | S5, LRU, Mamba, MinGRU | Efficient long sequences |
| Attention | SelfAttention, LinearAttention | Flexible, parallel training |

## RNNs

### LSTM / GRU

Standard recurrent networks using Flax cells:

```python
from flax import nnx
from memorax.networks import SequenceModelWrapper, RNN

# LSTM
lstm_torso = SequenceModelWrapper(RNN(nnx.LSTMCell(64, 64, rngs=nnx.Rngs(0))))

# GRU
gru_torso = SequenceModelWrapper(RNN(nnx.GRUCell(64, 64, rngs=nnx.Rngs(0))))
```

### xLSTM Variants

Extended LSTM architectures:

```python
from memorax.networks.sequence_models import sLSTM, mLSTM, xLSTM

# sLSTM (scalar LSTM)
slstm = sLSTM(hidden_dim=64)

# mLSTM (matrix LSTM)
mlstm = mLSTM(hidden_dim=64, head_dim=16, num_heads=4)

# xLSTM (combines sLSTM and mLSTM)
xlstm = xLSTM(hidden_dim=64)
```

## State Space Models

### LRU (Linear Recurrent Unit)

Efficient linear recurrence:

```python
from memorax.networks.sequence_models import LRU

lru = LRU(hidden_dim=64, state_dim=64)
```

### S5

Simplified Structured State Space:

```python
from memorax.networks.sequence_models import S5

s5 = S5(hidden_dim=64, state_dim=64)
```

### Mamba

Selective State Space Model:

```python
from memorax.networks.sequence_models import Mamba

mamba = Mamba(hidden_dim=64, state_dim=16, expand=2)
```

### MinGRU

Minimal GRU variant:

```python
from memorax.networks.sequence_models import MinGRU

mingru = MinGRU(hidden_dim=64)
```

## Attention

### Self-Attention

Standard multi-head attention:

```python
from memorax.networks.sequence_models import SelfAttention

attention = SelfAttention(
    num_heads=4,
    head_dim=16,
    use_rotary=True,  # Rotary position embeddings
)
```

### Linear Attention

Efficient linear-complexity attention:

```python
from memorax.networks.sequence_models import LinearAttention

linear_attention = LinearAttention(
    num_heads=4,
    head_dim=16,
    feature_map="elu",
)
```

## Memory Models

### FFM (Fast and Forgetful Memory)

```python
from memorax.networks.sequence_models import FFM

ffm = FFM(hidden_dim=64, memory_size=32)
```

### SHM (Stable Hadamard Memory)

```python
from memorax.networks.sequence_models import SHM

shm = SHM(hidden_dim=64, memory_size=32)
```

## Choosing a Model

### For Short Episodes (< 100 steps)
- **LSTM/GRU**: Simple and effective
- **sLSTM**: Enhanced gating

### For Long Episodes (100-1000 steps)
- **S5/LRU**: Efficient state space models
- **Mamba**: Selective attention to inputs

### For Very Long Episodes (> 1000 steps)
- **SelfAttention**: With positional embeddings
- **LinearAttention**: Linear complexity

### For Memory-Intensive Tasks
- **FFM/SHM**: Explicit memory mechanisms
- **mLSTM**: Matrix memory

## Example: Mamba Agent

```python
from memorax.algorithms import PPO, PPOConfig
from memorax.networks import Network, FeatureExtractor, MLP, SequenceModelWrapper, heads
from memorax.networks.sequence_models import Mamba

# Mamba-based actor
actor = Network(
    feature_extractor=FeatureExtractor(observation_extractor=MLP(features=(64,))),
    torso=SequenceModelWrapper(Mamba(hidden_dim=64)),
    head=heads.Categorical(action_dim=4),
)

# Mamba-based critic
critic = Network(
    feature_extractor=FeatureExtractor(observation_extractor=MLP(features=(64,))),
    torso=SequenceModelWrapper(Mamba(hidden_dim=64)),
    head=heads.VNetwork(),
)

agent = PPO(cfg, env, env_params, actor, critic, optimizer, optimizer)
```
