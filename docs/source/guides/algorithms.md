# Working with Algorithms

Memorax provides several RL algorithms optimized for memory-augmented learning.

## Available Algorithms

| Algorithm | Action Space | Use Case |
|-----------|--------------|----------|
| PPO | Discrete & Continuous | General-purpose, stable training |
| MAPPO | Discrete | Multi-agent PPO (independent policies) |
| DQN | Discrete | Value-based learning |
| SAC | Continuous | Maximum entropy RL |
| PQN | Discrete | On-policy Q-learning |
| R2D2 | Discrete | Recurrent value-based with prioritized replay |
| AC(Lambda) | Discrete | Online actor-critic with eligibility traces |
| GradientPPO | Discrete & Continuous | PPO with gradient eligibility traces |

## PPO (Proximal Policy Optimization)

Best for general-purpose training with memory architectures.

```python
from memorax.algorithms import PPO, PPOConfig

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
    burn_in_length=0,
)

agent = PPO(config, env, env_params, actor, critic, actor_optimizer, critic_optimizer)
```

### Key Parameters

- `num_envs`: Number of parallel environments for training
- `num_steps`: Steps per rollout before update
- `clip_coefficient`: PPO clipping coefficient (0.1-0.3)
- `burn_in_length`: Steps to warm up RNN hidden state before computing loss

## DQN (Deep Q-Network)

For discrete action spaces with value-based learning.

```python
from memorax.algorithms import DQN, DQNConfig

config = DQNConfig(
    num_envs=8,
    tau=1.0,
    target_update_frequency=1000,
    train_frequency=4,
    burn_in_length=0,
)

agent = DQN(config, env, env_params, q_network, optimizer)
```

## SAC (Soft Actor-Critic)

For continuous control with entropy regularization.

```python
from memorax.algorithms import SAC, SACConfig

config = SACConfig(
    num_envs=8,
    tau=0.005,
    train_frequency=1,
    target_update_frequency=1,
    target_entropy_scale=0.89,
    gradient_steps=1,
    burn_in_length=0,
)

agent = SAC(config, env, env_params, actor, critic, critic, actor_optimizer, critic_optimizer, alpha_optimizer)
```

## R2D2 (Recurrent Experience Replay in Distributed RL)

For discrete action spaces with recurrent networks and prioritized experience replay.

```python
from memorax.algorithms import R2D2, R2D2Config

config = R2D2Config(
    num_envs=8,
    tau=1.0,
    target_update_frequency=500,
    train_frequency=10,
    burn_in_length=10,
    sequence_length=80,
    n_step=5,
    priority_exponent=0.9,
    importance_sampling_exponent=0.6,
)

agent = R2D2(config, env, env_params, q_network, optimizer, buffer, epsilon_schedule, beta_schedule)
```

### Key Features

- **Prioritized Episode Replay**: Samples sequences weighted by TD-error priorities while respecting episode boundaries
- **N-step Returns**: Computes n-step temporal difference targets for better credit assignment
- **Burn-in**: Initializes hidden state context before computing losses
- **Double Q-learning**: Reduces overestimation bias using online network for action selection

## MAPPO (Multi-Agent PPO)

For multi-agent environments with independent policies.

```python
from memorax.algorithms import MAPPO, MAPPOConfig

config = MAPPOConfig(
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

agent = MAPPO(config, env, env_params, actor, critic, optimizer, optimizer)
```

## AC(Lambda) (Actor-Critic with Eligibility Traces)

Online actor-critic with lambda-weighted eligibility traces for true online learning.

```python
from memorax.algorithms import ACLambda, ACLambdaConfig

config = ACLambdaConfig(
    num_envs=8,
    trace_lambda=0.9,
    actor_lr=3e-4,
    critic_lr=1e-3,
    entropy_coefficient=0.01,
)

agent = ACLambda(config, env, env_params, actor, critic)
```

## GradientPPO

PPO variant with gradient eligibility traces for improved credit assignment in recurrent networks.

```python
from memorax.algorithms import GradientPPO, GradientPPOConfig

config = GradientPPOConfig(
    num_envs=8,
    num_steps=128,
    gae_lambda=0.95,
    num_minibatches=4,
    update_epochs=4,
    normalize_advantage=True,
    clip_coefficient=0.2,
    clip_value_loss=True,
    entropy_coefficient=0.01,
    regularization_coefficient=0.1,
    truncation_length=16,
)

agent = GradientPPO(config, env, env_params, actor, critic, optimizer, optimizer)
```

## Training Loop Pattern

All algorithms follow the same interface. Use `lox.spool` to capture logged metrics from `train` and `evaluate`:

```python
import lox

key, init_key = jax.random.split(key)
state = agent.init(init_key)
key, warmup_key = jax.random.split(key)
state = agent.warmup(warmup_key, state, num_steps=10_000)

train = lox.spool(agent.train)
key, train_key = jax.random.split(key)
state, logs = train(train_key, state, num_steps=100_000)

evaluate = lox.spool(agent.evaluate)
key, eval_key = jax.random.split(key)
state, eval_logs = evaluate(eval_key, state, num_steps=100)
```

## Burn-in for Recurrent Networks

When using RNNs/SSMs with off-policy algorithms (DQN, SAC, R2D2) or on-policy algorithms (PPO), use burn-in to establish hidden state context before computing losses:

```python
config = PPOConfig(burn_in_length=20, ...)
config = DQNConfig(burn_in_length=20, ...)
```

The first `burn_in_length` steps of each sequence are replayed without gradients to initialize the hidden state.
