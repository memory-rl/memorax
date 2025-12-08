from typing import Any, Protocol, Callable, TypeAlias

import jax
import flashbax as fbx
import gymnax

Key: TypeAlias = jax.Array
Array: TypeAlias = jax.Array

Buffer: TypeAlias = fbx.trajectory_buffer.TrajectoryBuffer
BufferState: TypeAlias = fbx.trajectory_buffer.TrajectoryBufferState
Environment: TypeAlias = gymnax.environments.environment.Environment
EnvParams: TypeAlias = gymnax.EnvParams
EnvState: TypeAlias = gymnax.EnvState

Carry: TypeAlias = Any
PyTree: TypeAlias = Any


class Logger(Protocol):
    def init(self, config: dict[str, Any]) -> Any: ...
    def log(self, state: Any, data: dict[str, Any], *, step: int) -> Any: ...
    def emit(self, state: Any) -> Any: ...
    def finish(self, state: Any) -> None: ...


class State(Protocol):
    """State of the algorithm."""

    step: int
    ...


class Algorithm(Protocol):
    init: Callable[[Key], tuple[Key, State]]
    warmup: Callable[[Key, State, int], tuple[Key, State]]
    train: Callable[[Key, State, int], tuple[Key, State, dict]]
    evaluate: Callable[[Key, State, int], tuple[Key, dict]]
