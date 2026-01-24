# Sequence Models

This guide covers the available sequence models for memory-augmented RL.

## Overview

Memorax supports three families of sequence models:

| Family | Models | Strengths |
|--------|--------|-----------|
| RNNs | LSTM, GRU, sLSTM, SHM | Simple, well-understood |
| State Space Models | S5, LRU, Mamba, MinGRU, mLSTM, FFM | Efficient long sequences |
| Attention | SelfAttention, LinearAttention | Flexible, parallel training |

## RNNs

### LSTM / GRU

Standard recurrent networks using Flax cells:

```python
import flax.linen as nn
from memorax.networks import RNN

# LSTM
lstm_torso = RNN(cell=nn.LSTMCell(features=64))

# GRU
gru_torso = RNN(cell=nn.GRUCell(features=64))
```

### sLSTM

Scalar LSTM with enhanced gating:

```python
from memorax.networks import RNN, sLSTMCell

slstm = RNN(cell=sLSTMCell(features=64))
```

### SHM (Stable Hadamard Memory)

```python
from memorax.networks import RNN, SHMCell

shm = RNN(cell=SHMCell(features=64, memory_size=32))
```

## State Space Models

All state space models use the `Memoroid` wrapper with their respective cells.

### LRU (Linear Recurrent Unit)

Efficient linear recurrence:

```python
from memorax.networks import Memoroid, LRUCell

lru = Memoroid(cell=LRUCell(features=64, hidden_dim=64))
```

### S5

Simplified Structured State Space:

```python
from memorax.networks import Memoroid, S5Cell

s5 = Memoroid(cell=S5Cell(features=64, state_dim=64))
```

### Mamba

Selective State Space Model:

```python
from memorax.networks import Memoroid, MambaCell

mamba = Memoroid(cell=MambaCell(features=64, num_heads=4, head_dim=16))
```

### MinGRU

Minimal GRU variant:

```python
from memorax.networks import Memoroid, MinGRUCell

mingru = Memoroid(cell=MinGRUCell(features=64))
```

### mLSTM (Matrix LSTM)

```python
from memorax.networks import Memoroid, mLSTMCell

mlstm = Memoroid(cell=mLSTMCell(features=64, num_heads=4, head_dim=16))
```

### FFM (Fast and Forgetful Memory)

```python
from memorax.networks import Memoroid, FFMCell

ffm = Memoroid(cell=FFMCell(features=64, memory_size=32))
```

## Attention

### Self-Attention

Standard multi-head attention (used directly, no wrapper needed):

```python
from memorax.networks import SelfAttention

attention = SelfAttention(
    features=64,
    num_heads=4,
    head_dim=16,
)
```

### Linear Attention

Efficient linear-complexity attention:

```python
from memorax.networks import Memoroid, LinearAttentionCell

linear_attention = Memoroid(cell=LinearAttentionCell(features=64, num_heads=4, head_dim=16))
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
from memorax.networks import Network, FeatureExtractor, MLP, Memoroid, MambaCell, heads

# Mamba-based actor
actor_network = Network(
    feature_extractor=FeatureExtractor(observation_extractor=MLP(features=(64,))),
    torso=Memoroid(cell=MambaCell(features=64, num_heads=4, head_dim=16)),
    head=heads.Categorical(action_dim=4),
)

# Mamba-based critic
critic_network = Network(
    feature_extractor=FeatureExtractor(observation_extractor=MLP(features=(64,))),
    torso=Memoroid(cell=MambaCell(features=64, num_heads=4, head_dim=16)),
    head=heads.VNetwork(),
)

agent = PPO(cfg, env, env_params, actor_network, critic_network, optimizer, optimizer)
```
