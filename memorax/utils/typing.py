from typing import Any, Protocol, TypeAlias

import flashbax as fbx
import gymnax
import jax

Key: TypeAlias = jax.Array
Array: TypeAlias = jax.Array

Buffer: TypeAlias = fbx.trajectory_buffer.TrajectoryBuffer
BufferState: TypeAlias = fbx.trajectory_buffer.TrajectoryBufferState
Environment: TypeAlias = gymnax.environments.environment.Environment
EnvParams: TypeAlias = gymnax.EnvParams
EnvState: TypeAlias = gymnax.EnvState
Discrete: TypeAlias = gymnax.environments.spaces.Discrete
Box: TypeAlias = gymnax.environments.spaces.Box

Carry: TypeAlias = Any
PyTree: TypeAlias = Any


class Logger(Protocol):
    def init(self, **kwargs) -> Any: ...
    def log(self, state: Any, data: dict[str, Any], step: Any, **kwargs) -> None: ...
    def emit(self, state: Any) -> None: ...
    def finish(self, state: Any) -> None: ...
