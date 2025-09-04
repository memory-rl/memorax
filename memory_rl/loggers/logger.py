from abc import ABC, abstractmethod
from functools import partial
from typing import Any, Generic, TypeVar, TypeAlias

import jax
import jax.numpy as jnp
import chex
from omegaconf import DictConfig

from memory_rl.utils.dataclasses import Transition

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

    @partial(jax.jit, static_argnames=("self",))
    def get_num_episodes(self, transitions) -> int:
        return transitions.done.sum()

    @partial(jax.jit, static_argnames=("self",))
    def get_episodic_lengths(self, transitions):
        done = transitions.done

        def step(carry_len, done_t):
            curr_len = carry_len + 1
            out = jnp.where(done_t, curr_len, jnp.zeros_like(curr_len))
            next_len = jnp.where(done_t, jnp.zeros_like(curr_len), curr_len)
            return next_len, out

        init_len = jnp.zeros_like(done[0], dtype=jnp.int32)
        _, lengths_at_done = jax.lax.scan(step, init_len, done)
        return lengths_at_done.sum() / self.get_num_episodes(transitions)

    @partial(jax.jit, static_argnames=("self",))
    def get_episodic_returns(self, transitions):
        r = transitions.reward
        done = transitions.done

        def step(carry_sum, inp):
            r_t, d_t = inp
            s = carry_sum + r_t
            out = jnp.where(d_t, s, jnp.zeros_like(s))
            next_s = jnp.where(d_t, jnp.zeros_like(s), s)
            return next_s, out

        init_sum = jnp.zeros_like(r[0])
        _, returns_at_done = jax.lax.scan(step, init_sum, (r, done))
        return returns_at_done.sum() / self.get_num_episodes(transitions)

    @partial(jax.jit, static_argnames=("self",))
    def get_discounted_episodic_returns(self, transitions, gamma: float):
        r = transitions.reward
        done = transitions.done
        gamma = jnp.asarray(gamma, dtype=r.dtype)

        def step(carry, inp):
            gsum, pow_ = carry
            r_t, d_t = inp
            gsum_new = gsum + r_t * pow_
            out = jnp.where(d_t, gsum_new, jnp.zeros_like(gsum_new))
            gsum_next = jnp.where(d_t, jnp.zeros_like(gsum_new), gsum_new)
            pow_next = jnp.where(d_t, jnp.ones_like(pow_), pow_ * gamma)
            return (gsum_next, pow_next), out

        init = (jnp.zeros_like(r[0]), jnp.ones_like(r[0]))
        _, disc_returns_at_done = jax.lax.scan(step, init, (r, done))
        return disc_returns_at_done.sum() / self.get_num_episodes(transitions)
