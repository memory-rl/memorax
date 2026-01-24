# Building Networks

This guide covers how to build neural networks for RL agents in Memorax.

## Network Architecture

A Memorax network consists of three components:

```
Input -> FeatureExtractor -> Torso -> Head -> Output
```

## Feature Extractor

The `FeatureExtractor` processes raw inputs into feature vectors:

```python
from memorax.networks import FeatureExtractor, MLP, CNN

# MLP for vector observations
feature_extractor = FeatureExtractor(
    observation_extractor=MLP(features=(64, 64)),
)

# CNN for image observations
feature_extractor = FeatureExtractor(
    observation_extractor=CNN(
        features=(32, 64, 64),
        kernel_sizes=(8, 4, 3),
        strides=(4, 2, 1),
    ),
)

# Include additional inputs
feature_extractor = FeatureExtractor(
    observation_extractor=MLP(features=(64,)),
    action_extractor=MLP(features=(32,)),
    reward_extractor=MLP(features=(16,)),
    done_extractor=MLP(features=(16,)),
)
```

## Torso (Sequence Models)

The torso processes temporal sequences. Wrap sequence models with `SequenceModelWrapper`:

```python
from memorax.networks import SequenceModelWrapper, RNN
from memorax.networks.sequence_models import Mamba, S5, LRU, SelfAttention

# LSTM
torso = SequenceModelWrapper(RNN(nnx.LSTMCell(64, 64, rngs=nnx.Rngs(0))))

# GRU
torso = SequenceModelWrapper(RNN(nnx.GRUCell(64, 64, rngs=nnx.Rngs(0))))

# Mamba (State Space Model)
torso = SequenceModelWrapper(Mamba(hidden_dim=64))

# S5 (State Space Model)
torso = SequenceModelWrapper(S5(hidden_dim=64, state_dim=64))

# Self-Attention
torso = SequenceModelWrapper(SelfAttention(num_heads=4, head_dim=16))
```

## Heads

Heads produce outputs for different RL objectives:

### Discrete Actions

```python
from memorax.networks import heads

# Categorical policy (discrete actions)
actor_head = heads.Categorical(action_dim=4)

# Q-network for DQN
q_head = heads.DiscreteQNetwork(action_dim=4)
```

### Continuous Actions

```python
# Gaussian policy
actor_head = heads.Gaussian(action_dim=2)

# Squashed Gaussian (bounded actions, used in SAC)
actor_head = heads.SquashedGaussian(action_dim=2)

# Q-network for continuous actions
q_head = heads.ContinuousQNetwork()
```

### Value Functions

```python
# Value network (critic)
critic_head = heads.VNetwork()
```

## Composing Networks

Use the `Network` class to compose components:

```python
from memorax.networks import Network, FeatureExtractor, MLP, SequenceModelWrapper, RNN, heads

# Actor network
actor = Network(
    feature_extractor=FeatureExtractor(observation_extractor=MLP(features=(64,))),
    torso=SequenceModelWrapper(RNN(nnx.LSTMCell(64, 64, rngs=nnx.Rngs(0)))),
    head=heads.Categorical(action_dim=4),
)

# Critic network
critic = Network(
    feature_extractor=FeatureExtractor(observation_extractor=MLP(features=(64,))),
    torso=SequenceModelWrapper(RNN(nnx.LSTMCell(64, 64, rngs=nnx.Rngs(1)))),
    head=heads.VNetwork(),
)
```

## Using Blocks

Add architectural blocks for more complex networks:

```python
from memorax.networks.blocks import Stack, PreNorm, Residual, FFN

# Stack multiple layers
torso = Stack(blocks=[
    PreNorm(Residual(SelfAttention(num_heads=4, head_dim=16))),
    PreNorm(Residual(FFN(hidden_dim=256))),
    PreNorm(Residual(SelfAttention(num_heads=4, head_dim=16))),
    PreNorm(Residual(FFN(hidden_dim=256))),
])
```

## Parameter Sharing

To share parameters between actor and critic:

```python
# Shared feature extractor and torso
shared_feature_extractor = FeatureExtractor(observation_extractor=MLP(features=(64,)))
shared_torso = SequenceModelWrapper(RNN(nnx.LSTMCell(64, 64, rngs=nnx.Rngs(0))))

actor = Network(shared_feature_extractor, shared_torso, heads.Categorical(action_dim=4))
critic = Network(shared_feature_extractor, shared_torso, heads.VNetwork())
```
