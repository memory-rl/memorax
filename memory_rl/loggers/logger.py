from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar, TypeAlias

import jax
import chex
from omegaconf import DictConfig

PyTree: TypeAlias = Any


@chex.dataclass(frozen=True)
class BaseLoggerState: ...


StateT = TypeVar("StateT", bound=BaseLoggerState)


@chex.dataclass(frozen=True)
class BaseLogger(Generic[StateT], ABC):

    @abstractmethod
    def init(self, cfg: DictConfig) -> StateT: ...

    @abstractmethod
    def log(self, state: StateT, data: PyTree, step: int) -> StateT: ...

    @abstractmethod
    def emit(self, state: StateT) -> StateT: ...

    def finish(self, state: StateT) -> None:
        return None


@chex.dataclass(frozen=True)
class LoggerState(BaseLoggerState):
    logger_states: tuple[BaseLoggerState, ...]


@chex.dataclass(frozen=True)
class Logger(BaseLogger[LoggerState]):
    loggers: tuple[BaseLogger[Any], ...]

    _is_leaf = staticmethod(lambda x: isinstance(x, (BaseLogger, BaseLoggerState)))

    def init(self, cfg: dict) -> LoggerState:
        logger_states = jax.tree.map(
            lambda logger: logger.init(cfg),
            self.loggers,
            is_leaf=self._is_leaf,
        )
        return LoggerState(logger_states=logger_states)

    def __post_init__(self):
        if not isinstance(self.loggers, tuple):
            object.__setattr__(self, "loggers", tuple(self.loggers.values()))

    def log(self, state: LoggerState, data: PyTree, step: int) -> LoggerState:
        logger_states = jax.tree.map(
            lambda logger, logger_state: logger.log(logger_state, data, step),
            self.loggers,
            state.logger_states,
            is_leaf=self._is_leaf,
        )
        return LoggerState(logger_states=logger_states)

    def emit(self, state: LoggerState) -> LoggerState:
        logger_states = jax.tree.map(
            lambda logger, logger_state: logger.emit(logger_state),
            self.loggers,
            state.logger_states,
            is_leaf=self._is_leaf,
        )
        return LoggerState(logger_states=logger_states)

    def finish(self, state: LoggerState) -> None:
        jax.tree.map(
            lambda logger, logger_state: logger.finish(logger_state),
            self.loggers,
            state.logger_states,
            is_leaf=self._is_leaf,
        )
