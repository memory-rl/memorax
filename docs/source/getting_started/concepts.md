# Core Concepts

## Architecture Overview

Memorax follows a modular architecture where agents are composed of an algorithm, networks, and an environment.

```{raw} html
<div style="font-family: monospace; margin: 2rem 0;">
  <div style="border: 2px solid var(--pst-color-border, #ccc); border-radius: 12px; padding: 1.5rem; background: var(--pst-color-surface, #fafafa);">
    <div style="font-weight: bold; font-size: 1.1rem; margin-bottom: 1rem; color: var(--pst-color-text-base, #333);">Agent</div>
    <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 1rem;">

      <div style="border: 2px solid var(--pst-color-border, #ccc); border-radius: 8px; padding: 1rem; text-align: center;">
        <div style="font-weight: bold; margin-bottom: 0.5rem; color: var(--pst-color-text-base, #333);">Algorithm</div>
        <div style="font-size: 0.85rem; color: var(--pst-color-text-muted, #666);">PPO, DQN, SAC, R2D2, ...</div>
        <div style="margin-top: 0.75rem; display: flex; gap: 0.4rem; justify-content: center; flex-wrap: wrap;">
          <code>Config</code>
          <code>State</code>
        </div>
      </div>

      <div style="border: 2px solid var(--pst-color-border, #ccc); border-radius: 8px; padding: 1rem;">
        <div style="font-weight: bold; margin-bottom: 0.5rem; text-align: center; color: var(--pst-color-text-base, #333);">Network</div>
        <div style="display: flex; flex-direction: column; gap: 0.3rem; align-items: center;">
          <code style="padding: 4px 12px;">Feature Extractor</code>
          <span style="color: var(--pst-color-text-muted, #999);">↓</span>
          <code style="padding: 4px 12px; white-space: nowrap;">Torso (Sequence Model)</code>
          <span style="color: var(--pst-color-text-muted, #999);">↓</span>
          <code style="padding: 4px 12px;">Head</code>
        </div>
      </div>

      <div style="border: 2px solid var(--pst-color-border, #ccc); border-radius: 8px; padding: 1rem; text-align: center;">
        <div style="font-weight: bold; margin-bottom: 0.5rem; color: var(--pst-color-text-base, #333);">Environment</div>
        <div style="font-size: 0.85rem; color: var(--pst-color-text-muted, #666);">Gymnax, Brax, Navix, ...</div>
        <div style="margin-top: 0.75rem; display: flex; gap: 0.4rem; justify-content: center; flex-wrap: wrap;">
          <code>env</code>
          <code>env_params</code>
        </div>
      </div>

    </div>
  </div>
</div>
```

## Algorithms

Every algorithm in Memorax follows the same pattern: a **Config** (frozen dataclass of hyperparameters), a **State** (all mutable training state: parameters, optimizer state, hidden carries), and the **Algorithm** class itself which provides `init()`, `train()`, `warmup()`, and `evaluate()`. You create an agent by passing the config, environment, networks, and optimizers, then call `init` to get the initial state and `train` to run updates:

```python
from memorax.algorithms import PPO, PPOConfig

config = PPOConfig(
    num_envs=8, num_steps=128, gae_lambda=0.95, num_minibatches=4,
    update_epochs=4, normalize_advantage=True, clip_coefficient=0.2,
    clip_value_loss=True, entropy_coefficient=0.01,
)

agent = PPO(config, env, env_params, actor, critic, optimizer, optimizer)
key, state = agent.init(key)
key, state = agent.train(key, state, num_steps=1000)
```

All algorithms share this interface, so switching from PPO to DQN or SAC is just a matter of changing the config and providing the right networks.

## Networks

Networks follow the `FeatureExtractor → Torso → Head` pipeline shown above. The **feature extractor** processes raw observations into a feature vector. Any Flax module works here. The **torso** handles temporal modeling: RNNs and Memoroid cells work directly as sequence models, while non-recurrent modules need a `SequenceModelWrapper`. The **head** produces the algorithm's output: action distributions for policy methods, Q-values for value methods, or state values for critics:

```python
import flax.linen as nn
from memorax.networks import FeatureExtractor, Network, RNN, Memoroid, MambaCell, heads

feature_extractor = FeatureExtractor(
    observation_extractor=nn.Sequential((nn.Dense(64), nn.relu)),
)

actor = Network(feature_extractor, torso=RNN(cell=nn.GRUCell(features=64)), head=heads.Categorical(action_dim=4))
critic = Network(feature_extractor, torso=Memoroid(cell=MambaCell(features=64)), head=heads.VNetwork())
```

See the {doc}`../guides/networks` guide for the full list of available torsos, heads, and composable blocks.

## JAX Patterns

Memorax is built on JAX, which brings a few patterns that differ from typical PyTorch workflows. **Vectorized environments**: all training runs `num_envs` environments in parallel, fully JIT-compiled. **Explicit random state**: JAX uses functional random keys that you thread through calls. `agent.init` and `agent.train` both consume and return a key. **JIT compilation**: the first call to `train` may be slow as JAX traces and compiles the computation graph, but subsequent calls are fast.

```python
key = jax.random.key(0)
key, state = agent.init(key)
key, state = agent.train(key, state, num_steps=1000)
```

## Logging

Memorax doesn't return metrics from `train()` directly. Instead, wrap your training function with `lox.spool` to capture logged values, and wrap the environment with `RecordEpisodeStatistics` to track episode returns and lengths. Use `jax.vmap` to vectorize across random seeds:

```python
import lox
from memorax.environments.wrappers import RecordEpisodeStatistics

env = RecordEpisodeStatistics(env)
train = jax.vmap(lox.spool(agent.train), in_axes=(0, 0, None))
(keys, state), logs = train(keys, state, num_steps)
```

Memorax provides several logger backends: `DashboardLogger` for a live terminal UI, `WandbLogger`, `TensorBoardLogger`, `FileLogger`, and `CheckpointLogger`. Combine multiple backends with the `MultiLogger` wrapper.

## Buffers

Off-policy algorithms like DQN and R2D2 need replay buffers. Memorax provides episode-aware buffers built on Flashbax that sample complete sequences while respecting episode boundaries. The basic `make_episode_buffer` handles uniform sampling, while `make_prioritised_episode_buffer` adds TD-error-weighted sampling with importance correction. This is what R2D2 uses for prioritized experience replay:

```python
from memorax.buffers import make_episode_buffer, make_prioritised_episode_buffer

buffer = make_episode_buffer(
    max_length=100_000, min_length=1000,
    sample_batch_size=32, sample_sequence_length=16, add_batch_size=8,
)

prioritised_buffer = make_prioritised_episode_buffer(
    max_length=100_000, min_length=1000,
    sample_batch_size=32, sample_sequence_length=16, add_batch_size=8,
    priority_exponent=0.6,
)
```
