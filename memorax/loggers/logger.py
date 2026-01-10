from abc import ABC, abstractmethod
from typing import Any, Generic, TypeAlias, TypeVar

import jax
import jax.numpy as jnp
from flax import struct

from memorax.utils.typing import Array

PyTree: TypeAlias = Any


@struct.dataclass(frozen=True)
class BaseLoggerState: ...


StateT = TypeVar("StateT", bound=BaseLoggerState)


@struct.dataclass(frozen=True)
class BaseLogger(Generic[StateT], ABC):

    @abstractmethod
    def init(self, **kwargs) -> StateT: ...

    @abstractmethod
    def log(self, state: StateT, data: PyTree, step: PyTree) -> StateT: ...

    @abstractmethod
    def emit(self, state: StateT) -> StateT: ...

    def finish(self, state: StateT) -> None:
        return None


@struct.dataclass(frozen=True)
class LoggerState(BaseLoggerState):
    logger_states: dict[str, BaseLoggerState]


@struct.dataclass(frozen=True)
class Logger(BaseLogger[LoggerState]):
    loggers: dict[str, BaseLogger[Any]] | list[BaseLogger[Any]]

    _is_leaf = staticmethod(lambda x: isinstance(x, (BaseLogger, BaseLoggerState)))

    def init(self, **kwargs) -> LoggerState:
        logger_states = jax.tree.map(
            lambda logger: logger.init(**kwargs),
            self.loggers,
            is_leaf=self._is_leaf,
        )
        return LoggerState(logger_states=logger_states)

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

    @staticmethod
    def get_episode_statistics(transitions, prefix: str):
        def describe(a: Array, metric: str):
            return {
                f"{prefix}/mean_{metric}": jnp.nanmean(a),
                f"{prefix}/std_{metric}": jnp.nanstd(a),
            }

        num_episodes = {f"{prefix}/num_episodes": transitions.num_episodes}
        episode_lengths = describe(transitions.episode_lengths, "episode_lengths")
        episode_returns = describe(transitions.episode_returns, "episode_returns")

        return {
            **num_episodes,
            **episode_lengths,
            **episode_returns,
        }
