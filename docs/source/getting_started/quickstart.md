# Quick Start

This guide walks through training a PPO agent with GRU memory on CartPole.

## Environment

Create a Gymnax environment wrapped with episode statistics tracking:

```python
from memorax.environments import make
from memorax.environments.wrappers import RecordEpisodeStatistics

env, env_params = make("gymnax::CartPole-v1")
env = RecordEpisodeStatistics(env)
```

## Configuration

Define the PPO hyperparameters:

```python
from memorax.algorithms import PPOConfig

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
```

## Networks

Memorax networks follow a `feature_extractor → torso → head` pipeline. The feature extractor processes raw observations, the torso handles temporal sequence modeling, and the head produces the final output (action distribution or value estimate).

```python
import flax.linen as nn
from memorax.networks import FeatureExtractor, Network, RNN, heads

d_model = 64

feature_extractor = FeatureExtractor(
    observation_extractor=nn.Sequential((nn.Dense(d_model), nn.relu)),
)
torso = RNN(cell=nn.GRUCell(features=d_model))

actor_network = Network(
    feature_extractor=feature_extractor,
    torso=torso,
    head=heads.Categorical(
        action_dim=env.action_space(env_params).n,
        kernel_init=nn.initializers.orthogonal(scale=0.01),
    ),
)
critic_network = Network(
    feature_extractor=feature_extractor,
    torso=torso,
    head=heads.VNetwork(kernel_init=nn.initializers.orthogonal(scale=1.0)),
)
```

## Agent

Combine the config, environment, networks, and optimizer into a PPO agent:

```python
import optax
from memorax.algorithms import PPO

optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adam(learning_rate=3e-4, eps=1e-5),
)

agent = PPO(
    config=config,
    env=env,
    env_params=env_params,
    actor_network=actor_network,
    critic_network=critic_network,
    actor_optimizer=optimizer,
    critic_optimizer=optimizer,
)
```

## Training

Use `jax.vmap` to vectorize across seeds and `lox.spool` to capture training metrics. The logger displays a live dashboard in the terminal:

```python
from dataclasses import asdict

import jax
import lox
from memorax.loggers import DashboardLogger, MultiLogger

logger = MultiLogger([DashboardLogger(total_timesteps=500_000, summary={"Algorithm": "PPO-GRU", "Environment": "CartPole"})])

init = jax.vmap(agent.init)
train = jax.vmap(lox.spool(agent.train), in_axes=(0, 0, None))

key = jax.random.key(0)
num_seeds = 1
key, init_key = jax.random.split(key)
state = init(jax.random.split(init_key, num_seeds))

for i in range(0, 500_000, 10_000):
    key, train_key = jax.random.split(key)
    state, logs = train(jax.random.split(train_key, num_seeds), state, 10_000)

    info = logs.pop("info")
    episode_returns = info["returned_episode_returns"][info["returned_episode"]]
    episode_lengths = info["returned_episode_lengths"][info["returned_episode"]]

    data = {
        "training/episode_returns": episode_returns,
        "training/episode_lengths": episode_lengths,
        **logs,
    }
    logger.log(data, step=state.step.mean().item())

logger.finish()
```

## Next Steps

- Learn about different {doc}`../guides/algorithms`
- Explore available {doc}`../guides/sequence_models`
- Build custom {doc}`../guides/networks`
