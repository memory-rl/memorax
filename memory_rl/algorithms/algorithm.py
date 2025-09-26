from typing import Callable, Protocol

import chex

from memory_rl.algorithms import (
    dqn,
    drqn,
    ppo,
    ppo_continuous,
    rppo,
    rppo_continuous,
    sac,
    rsac,
    sacd,
    rsacd,
    pqn,
    rpqn,
)

register = {
    "dqn": dqn.make,
    "drqn": drqn.make,
    "ppo": ppo.make,
    "ppo_continuous": ppo_continuous.make,
    "rppo": rppo.make,
    "rppo_continuous": rppo_continuous.make,
    "sac": sac.make,
    "rsac": rsac.make,
    "sacd": sacd.make,
    "rsacd": rsacd.make,
    "pqn": pqn.make,
    "rpqn": rpqn.make,
}


class State(Protocol):
    """State of the algorithm."""

    step: int
    ...


class Algorithm(Protocol):
    init: Callable[[chex.PRNGKey], tuple[chex.PRNGKey, State]]
    warmup: Callable[[chex.PRNGKey, State, int], tuple[chex.PRNGKey, State]]
    train: Callable[[chex.PRNGKey, State, int], tuple[chex.PRNGKey, State, dict]]
    evaluate: Callable[[chex.PRNGKey, State, int], tuple[chex.PRNGKey, dict]]


def make(algorithm_id: str, cfg, env, env_params) -> Algorithm:

    if algorithm_id not in register:
        raise ValueError(f"Unknown algorithm {algorithm_id}")

    return register[algorithm_id](cfg, env, env_params)
