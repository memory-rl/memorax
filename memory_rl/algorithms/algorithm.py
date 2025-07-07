from typing import Callable, Protocol

import chex

from memory_rl.algorithms import (
    make_dqn,
    make_drqn,
    make_ppo,
    make_rppo,
    make_rppo_continuous,
    make_rsac,
    make_sac,
    make_sacd,
)

register = {
    "dqn": make_dqn,
    "drqn": make_drqn,
    "ppo": make_ppo,
    "rppo": make_rppo,
    "rppo_continuous": make_rppo_continuous,
    "sac": make_sac,
    "rsac": make_rsac,
    "sacd": make_sacd,
}


class State(Protocol):
    """State of the algorithm."""

    step: int
    ...


class Algorithm(Protocol):
    init: Callable[[chex.PRNGKey], tuple[chex.PRNGKey, State]]
    warmup: Callable[[chex.PRNGKey, State], tuple[chex.PRNGKey, State]]
    train: Callable[[chex.PRNGKey, State, int], tuple[chex.PRNGKey, State, dict]]
    evaluate: Callable[[chex.PRNGKey, State, int], tuple[chex.PRNGKey, dict]]


def make(algorithm_id: str, cfg, env, env_params) -> Algorithm:

    if algorithm_id not in register:
        raise ValueError(f"Unknown algorithm {algorithm_id}")

    return register[algorithm_id](cfg, env, env_params)
