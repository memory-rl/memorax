"""Benchmark PQN on PopJym with performax profiling.

Usage:
    uv run benchmark.py algorithm=pqn environment=popjym/autoencode/easy torso=mlp
"""

import hydra
import jax
from hydra.utils import instantiate
from omegaconf import OmegaConf
from performax import enable_barriers, profile, track

from memorax.algorithms.pqn import PQN
from src import algorithm, environment

# Patch PQN methods with @track before any instances are created
TRACKED_METHODS = [
    "_greedy_action",
    "_epsilon_greedy_action",
    "_step",
    "_learn",
    "_update_epoch",
    "_update_minibatch",
]

enable_barriers()
for method_name in TRACKED_METHODS:
    original = getattr(PQN, method_name)
    tracked = track(original, name=method_name.lstrip("_"))
    setattr(PQN, method_name, tracked)


@hydra.main(version_base=None, config_path="./config", config_name="config")
def main(cfg):
    env, env_params = environment.make(**cfg.environment)
    agent = algorithm.make(cfg, env, env_params)

    key = jax.random.key(cfg.seed)
    keys = jax.random.split(key, cfg.num_seeds)

    init = jax.vmap(agent.init)
    train_fn = jax.vmap(agent.train, in_axes=(0, 0, None))
    profiled_train = profile(train_fn)

    keys, state = init(keys)

    # Warmup (JIT compile)
    print("Compiling...")
    _ = jax.block_until_ready(train_fn(keys, state, cfg.num_train_steps))

    # Profiled run
    print("Profiling...")
    (keys, state, transitions), stats = profiled_train(keys, state, cfg.num_train_steps)

    print()
    print(f"Config: {cfg.algorithm.name} | {cfg.environment.env_id} | {OmegaConf.to_container(cfg.torso)}")
    print(f"  num_envs={cfg.algorithm.num_envs} num_steps={cfg.algorithm.num_steps} "
          f"minibatches={cfg.algorithm.num_minibatches} epochs={cfg.algorithm.update_epochs}")
    print()
    print(stats)


if __name__ == "__main__":
    main()
