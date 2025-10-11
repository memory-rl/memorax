# Memory-RL — Modular Recurrent RL in JAX/Flax

*A plug-and-play reinforcement-learning stack that pairs standard algorithms with configurable memory, unified environments, and composable logging.*

---

## Why this library?

Modern RL lives under partial observability. State lives in memory. You want to swap `sLSTM ↔ GTrXL ↔ GPT2-cell` without rewriting your training loop, run the same code on `gymnax`, `brax`, or `mujoco`, and send metrics to any logger you like. That’s exactly what this repo does—clean interfaces, typed configs/states, fast JAX scans, and recurrent variants of the usual suspects.

---

## Highlights

* **Memory is a first-class citizen.** Drop-in recurrent cells (`sLSTM`, `mLSTM`, `xLSTM`, `GTrXL`, `GPT2`-style, `S5`, `SHM`, `DNC`, `FFM`) with a `MaskedRNN` wrapper (burn-in, carry slicing/detach).
* **Recurrent algorithms.** `R*` counterparts of major methods for POMDPs (`DRQN`, `RPPO`, `RPQN`, `RSAC`, `RSACD`, …).
* **Unified environments.** One factory, consistent spaces, identical `reset/step` signatures across `gymnax`, `brax`, `mujoco`, `craftax`, `navix`, `gxm`, `popgym_arcade`, `popjaxrl`.
* **Pluggable logging.** Console/File, TensorBoard, Weights & Biases, Neptune, and a dashboard aggregator—same `Logger` API everywhere.
* **Typed, minimal loop.** Every algorithm exposes `init`, `warmup`, `train`, `evaluate` with dataclass `Config` and `State`.

---

## Package layout

```
algorithms/   # training loops + configs for value/policy methods
networks/     # MLP/CNN, heads, (Recurrent)Network orchestration + memory cells
buffers/      # rollout storage (episode/trajectory via flashbax)
environments/ # factory + wrappers (+ dated runs in environments/outputs/YYYY-MM-DD/)
loggers/      # backends + composition
utils/        # GAE, target updates, Transition, decorators, typing protocol
```

### Algorithms

**Discrete:** `DQN`, `DRQN`, `PPO`, `RPPO`, `PQN`, `RPQN`, `SACD`, `RSACD`
**Continuous:** `PPOContinuous`, `RPPOContinuous`, `SAC`, `RSAC`

All implement:

```python
init(config, rng) -> State
warmup(state, env, logger) -> State
train(state, env, logger, num_steps) -> (State, Metrics)
evaluate(state, env, episodes) -> EvalMetrics
```

### Networks

* **Feed-forward:** `MLP`, `CNN`, policy/value heads.
* **Memory cells:** `sLSTM`, `mLSTM`, `xLSTM`, `GTrXL`, `GPT2`-style cell, `S5`, `SHM`, `DNC`, `FFM`.
* **Wrapper:** `MaskedRNN` supports burn-in, variable-length sequences, carry slicing/detach, and shared/separate feature extractors.

### Buffers

* `episode_buffer.py` built on **flashbax** trajectory buffers (episode/trajectory handling, batched sampling, masks).

### Environments

* Factory: `environments.environment.make("namespace::env_id")`
* Namespaces: `gymnax`, `brax`, `mujoco`, `craftax`, `navix`, `gxm`, `popgym_arcade`, `popjaxrl`
* Extras: observation masking utilities.

### Loggers

Console, File, **TensorBoard**, **Weights & Biases**, **Neptune**, and a simple dashboard aggregator. Compose multiple backends via a common `Logger` interface.

---

## Quickstart

### Install

```bash
pip install memory-rl
```

### Train your first recurrent agent

```python
import jax, jax.numpy as jnp
import flax.linen as nn
from memory_rl.algorithms.rppo import RPPOConfig, RPPO
from memory_rl.environments import environment
from memory_rl.loggers import Logger, ConsoleLogger, TensorBoardLogger
from memory_rl.networks import MLP, RecurrentNetwork, heads, SharedFeatureExtractor

env, env_params = environment.make("gymnax::CartPole-v1")

cfg = RPPOConfig(
   name="rppo",
   learning_rate=3e-4,
   num_envs=32,
   num_eval_envs=16,
   num_steps=64,
   anneal_lr=True,
   gamma=0.99,
   gae_lambda=0.95,
   num_minibatches=4,
   update_epochs=4,
   normalize_advantage=True,
   clip_coef=0.2,
   clip_vloss=True,
   ent_coef=0.0,
   vf_coef=0.5,
   max_grad_norm=0.5,
   learning_starts=0,
)

actor = RecurrentNetwork(
   feature_extractor=SharedFeatureExtractor(extractor=MLP(features=(128,))),
   torso=nn.GRUCell(features=128),
   head=heads.Categorical(
      action_dim=env.action_space(env_params).n,
      kernel_init=nn.initializers.orthogonal(scale=0.01),
   ),
)
actor_optimizer = optax.chain(
  optax.clip_by_global_norm(cfg.max_grad_norm),
  optax.adam(learning_rate=learning_rate, eps=1e-5),
)

critic = RecurrentNetwork(
   feature_extractor=SharedFeatureExtractor(extractor=MLP(features=(128,))),
   torso=nn.GRUCell(features=(128,)),
   head=heads.VNetwork(
      kernel_init=nn.initializers.orthogonal(scale=0.01),
   ),
)
critic_optimizer = optax.chain(
   optax.clip_by_global_norm(rppo_config.max_grad_norm),
   optax.adam(learning_rate=learning_rate, eps=1e-5),
)


logger = Logger([ConsoleLogger(), TensorBoardLogger(log_dir="runs/cartpole_rppo")])
logger_state = logger.init(cfg)

key = jax.random.key(0)

agent = RPPO(
   cfg=cfg,
   env=env,
   env_params=env_params,
   actor=actor,
   actor_optimizer=actor_optimizer,
   critic=critic,
   critic_optimizer=critic_optimizer,
)

key, state = agent.init(key)
key, state, transitions = agent.train(key, state, num_steps=500_000)

training_statistics = Logger.get_episode_statistics(transitions, cfg.gamma, "training")
losses = Logger.get_losses(transitions)
data = {**losses, **episode_statistics}
logger_state = logger.log(logger_state, data, step=state.step)

key, state, transitions = agent.evaluate(key, state, num_steps=5_000)
evaluation_statistics = Logger.get_episode_statistics(transitions, cfg.gamma, "evaluation")
logger_state = logger.log(logger_state, evaluation_statistics, step=state.step)
```

---

## Environment coverage

```python
# Examples
make("gymnax::CartPole-v1")
make("brax::ant")
make("mujoco::HalfCheetah-v4")
make("craftax::MiniGrid-DoorKey-8x8-v0")
make("navix::FourRooms")
make("gxm::Gymnasium/ALE/Breakout-v5")
make("popgym_arcade::SpaceInvaders")
make("popjaxrl::Catch")
```

* Observation masking utilities for partial observability experiments.

---

## Supported methods (at a glance)

| Family            | Discrete                              | Continuous                            |
| ----------------- | ------------------------------------- | ------------------------------------- |
| Value-based       | `DQN`, **`DRQN`**, `PQN, **`RPQN`**   | —                                     |
| Actor-critic      | `PPO`, **`RPPO`**                     | `PPOContinuous`, **`RPPOContinuous`** |
| Soft actor-critic | `SACD`, **`RSACD`**                   | `SAC`, **`RSAC`**                     |
| Others            | `PQN`, **`RPQN`**                     | -                                     |

**Bold** = recurrent variant available.

---

## Contributing

Issues and PRs welcome. Keep PRs focused, typed, and covered:

* Add/extend an algorithm: implement the protocol + a minimal example.
* Add a memory cell: register in `networks/` and ensure `MaskedRNN` tests pass.
* Add an environment: implement the factory adapter and space spec.
* Add a logger: implement the backend state and register with the composite.


## Citation

If this library helps your research, cite it like so:

```bibtex
@software{memjaxrl,
  title        = {Memory-RL: Modular Recurrent Reinforcement Learning in JAX},
  year         = {2025},
  author       = {Noah Farr},
  url          = {https://github.com/memory-r/memory-rl}
}
```

---
