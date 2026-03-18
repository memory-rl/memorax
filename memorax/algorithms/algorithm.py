from typing import Callable, Protocol

from memorax.utils.typing import Key


class State(Protocol):
    step: int
    ...


class Algorithm(Protocol):
    init: Callable[[Key], State]
    warmup: Callable[[Key, State, int], State]
    train: Callable[[Key, State, int], State]
    evaluate: Callable[[Key, State, int], State]
