# Working with Algorithms

Memorax provides several RL algorithms optimized for memory-augmented learning.

## Available Algorithms

| Algorithm | Action Space | Use Case |
|-----------|--------------|----------|
| PPO | Discrete & Continuous | General-purpose, stable training |
| IPPO | Discrete | Multi-agent PPO (independent policies) |
| DQN | Discrete | Value-based learning |
| SAC | Continuous | Maximum entropy RL |
| PQN | Discrete | On-policy Q-learning |
| R2D2 | Discrete | Recurrent value-based learning with prioritized replay |

## PPO (Proximal Policy Optimization)

Best for general-purpose training with memory architectures.

```python
from memorax.algorithms import PPO, PPOConfig

cfg = PPOConfig(
    name="PPO-Experiment",
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
    burn_in_length=0,  # RNN burn-in steps
)

agent = PPO(cfg, env, env_params, actor, critic, actor_optimizer, critic_optimizer)
```

### Key Parameters

- `num_envs`: Number of parallel environments for training
- `num_steps`: Steps per rollout before update
- `clip_coef`: PPO clipping coefficient (0.1-0.3)
- `burn_in_length`: Steps to "warm up" RNN hidden state

## DQN (Deep Q-Network)

For discrete action spaces with value-based learning.

```python
from memorax.algorithms import DQN, DQNConfig

cfg = DQNConfig(
    name="DQN-Experiment",
    num_envs=8,
    buffer_size=100_000,
    batch_size=32,
    learning_starts=1000,
    target_update_freq=1000,
    gamma=0.99,
    epsilon_start=1.0,
    epsilon_end=0.05,
    epsilon_decay_steps=50_000,
    burn_in_length=0,  # RNN burn-in steps
)

agent = DQN(cfg, env, env_params, q_network, optimizer)
```

## SAC (Soft Actor-Critic)

For continuous control with entropy regularization.

```python
from memorax.algorithms import SAC, SACConfig

cfg = SACConfig(
    name="SAC-Experiment",
    num_envs=8,
    buffer_size=1_000_000,
    batch_size=256,
    learning_starts=10_000,
    gamma=0.99,
    tau=0.005,
    alpha=0.2,  # Temperature parameter
    auto_alpha=True,  # Learn temperature
    burn_in_length=0,  # RNN burn-in steps
)

agent = SAC(cfg, env, env_params, actor, critic, critic, actor_optimizer, critic_optimizer, alpha_optimizer)
```

## R2D2 (Recurrent Experience Replay in Distributed RL)

For discrete action spaces with recurrent networks and prioritized experience replay.

```python
from memorax.algorithms import R2D2, R2D2Config
from memorax.buffers import make_prioritised_episode_buffer

cfg = R2D2Config(
    name="R2D2-Experiment",
    learning_rate=3e-4,
    num_envs=8,
    num_eval_envs=16,
    buffer_size=100_000,
    gamma=0.99,
    tau=1.0,
    target_network_frequency=500,
    batch_size=32,
    start_e=1.0,
    end_e=0.05,
    exploration_fraction=0.5,
    learning_starts=10_000,
    train_frequency=10,
    burn_in_length=10,     # Steps to warm up RNN hidden state
    sequence_length=80,    # Sequence length for training
    n_step=5,              # N-step returns
    priority_exponent=0.9, # PER priority exponent (alpha)
    importance_sampling_exponent=0.6,  # PER IS exponent (beta)
    double=True,           # Use Double Q-learning
)

buffer = make_prioritised_episode_buffer(
    max_length=cfg.buffer_size,
    min_length=cfg.batch_size * cfg.sequence_length,
    sample_batch_size=cfg.batch_size,
    sample_sequence_length=cfg.sequence_length,
    add_batch_size=cfg.num_envs,
    add_sequences=True,
    priority_exponent=cfg.priority_exponent,
)

epsilon_schedule = optax.linear_schedule(cfg.start_e, cfg.end_e, exploration_steps)
beta_schedule = optax.linear_schedule(cfg.importance_sampling_exponent, 1.0, total_steps)

agent = R2D2(cfg, env, env_params, q_network, optimizer, buffer, epsilon_schedule, beta_schedule)
```

### Key Features

- **Prioritized Episode Replay**: Samples sequences weighted by TD-error priorities while respecting episode boundaries
- **N-step Returns**: Computes n-step temporal difference targets for better credit assignment
- **Burn-in**: Initializes hidden state context before computing losses
- **Double Q-learning**: Reduces overestimation bias using online network for action selection

## Training Loop Pattern

All algorithms follow the same interface:

```python
# Initialize
key, state = agent.init(key)

# Optional: warmup (fill replay buffer for off-policy)
key, state = agent.warmup(key, state, num_steps=10_000)

# Train
key, state, transitions = agent.train(key, state, num_steps=100_000)

# Evaluate
key, returns = agent.evaluate(key, state, num_episodes=10)
```

## Burn-in for Recurrent Networks

When using RNNs/SSMs with off-policy algorithms (DQN, SAC) or on-policy algorithms (PPO), use burn-in to establish hidden state context before computing losses. This is especially important when sampling sequences from a replay buffer, as the initial hidden state is unknown.

```python
# Off-policy (DQN, SAC)
cfg = DQNConfig(
    burn_in_length=20,  # 20 steps of context before learning
    # ...
)

cfg = SACConfig(
    burn_in_length=20,
    # ...
)

# On-policy (PPO)
cfg = PPOConfig(
    burn_in_length=20,
    num_steps=128,
)
```

The first `burn_in_length` steps of each sequence are replayed without gradients to initialize the hidden state. Only the remaining steps are used for computing losses and updates.
