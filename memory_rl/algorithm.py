from typing import Callable, Protocol

import chex


class State(Protocol):
    """State of the algorithm."""

    ...


class Algorithm(Protocol):
    init: Callable[[chex.PRNGKey], State]
    train: Callable[[chex.PRNGKey, State, int], tuple[chex.PRNGKey, State, dict]]
    evaluate: Callable[[chex.PRNGKey, State, int], tuple[chex.PRNGKey, dict]]
