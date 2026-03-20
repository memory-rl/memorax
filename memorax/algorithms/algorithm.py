from typing import Callable, Protocol, TypeVar

from memorax.utils.typing import Key

State = TypeVar("State")

class Algorithm(Protocol[State]):
    init: Callable[[Key], State]
    warmup: Callable[[Key, State, int], State]
    train: Callable[[Key, State, int], State]
    evaluate: Callable[[Key, State, int], State]