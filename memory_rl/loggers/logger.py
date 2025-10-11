from abc import ABC, abstractmethod
from functools import partial
from typing import Any, Generic, TypeVar, TypeAlias

import jax
import jax.numpy as jnp
from flax import struct

PyTree: TypeAlias = Any


@struct.dataclass(frozen=True)
class BaseLoggerState: ...


StateT = TypeVar("StateT", bound=BaseLoggerState)


@struct.dataclass(frozen=True)
class BaseLogger(Generic[StateT], ABC):

    @abstractmethod
    def init(self, cfg) -> StateT: ...

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
    loggers: dict[str, BaseLogger[Any]]

    _is_leaf = staticmethod(lambda x: isinstance(x, (BaseLogger, BaseLoggerState)))

    def init(self, cfg: dict) -> LoggerState:
        logger_states = jax.tree.map(
            lambda logger: logger.init(cfg),
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
    @partial(jax.jit, static_argnames=("prefix",))
    def get_episode_statistics(transitions, gamma: float, prefix: str):
        return {
            f"{prefix}/num_episodes": Logger.get_num_episodes(transitions),
            f"{prefix}/mean_episodic_lengths": Logger.get_episodic_lengths(transitions),
            f"{prefix}/mean_episodic_returns": Logger.get_episodic_returns(transitions),
            f"{prefix}/mean_discounted_episodic_returns": Logger.get_discounted_episodic_returns(
                transitions, gamma
            ),
            f"{prefix}/iqm_episodic_returns": Logger.get_iqm_episodic_returns(
                transitions
            ),
            f"{prefix}/iqm_discounted_episodic_returns": Logger.get_iqm_discounted_episodic_returns(
                transitions, gamma
            ),
            f"{prefix}/iqm_episodic_lengths": Logger.get_iqm_episodic_lengths(
                transitions
            ),
            f"{prefix}/mmer_episodic_returns": Logger.get_mmer_episodic_returns(
                transitions
            ),
            f"{prefix}/mmer_discounted_episodic_returns": Logger.get_mmer_discounted_episodic_returns(
                transitions, gamma
            ),
            f"{prefix}/mmer_episodic_lengths": Logger.get_mmer_episodic_lengths(
                transitions
            ),
        }

    @staticmethod
    @jax.jit
    def get_num_episodes(transitions) -> int:
        return transitions.done.sum()

    @staticmethod
    @jax.jit
    def get_episodic_lengths(transitions):
        done = transitions.done

        def step(carry_len, done_t):
            curr_len = carry_len + 1
            out = jnp.where(done_t, curr_len, jnp.zeros_like(curr_len))
            next_len = jnp.where(done_t, jnp.zeros_like(curr_len), curr_len)
            return next_len, out

        init_len = jnp.zeros_like(done[0], dtype=jnp.int32)
        _, lengths_at_done = jax.lax.scan(step, init_len, done)
        return lengths_at_done.sum() / Logger.get_num_episodes(transitions)

    @staticmethod
    @jax.jit
    def get_episodic_returns(transitions):
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
        return returns_at_done.sum() / Logger.get_num_episodes(transitions)

    @staticmethod
    @jax.jit
    def get_discounted_episodic_returns(transitions, gamma: float):
        r = transitions.reward
        done = transitions.done

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
        return disc_returns_at_done.sum() / Logger.get_num_episodes(transitions)

    @staticmethod
    def get_losses(transitions):
        return {
            k: v.mean() for k, v in transitions.info.items() if k.startswith("losses")
        }

    @staticmethod
    @jax.jit
    def get_iqm_episodic_returns(transitions):
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

        vals = jnp.where(done, returns_at_done, jnp.nan).reshape(-1)

        q1, q3 = jnp.nanquantile(vals, jnp.array([0.25, 0.75]))
        in_iqr = (~jnp.isnan(vals)) & (vals >= q1) & (vals <= q3)
        count = in_iqr.sum()
        total = jnp.where(in_iqr, vals, 0.0).sum()
        return jnp.where(count > 0, total / count, jnp.array(0.0, dtype=vals.dtype))

    @staticmethod
    @jax.jit
    def get_iqm_discounted_episodic_returns(transitions, gamma: float):
        r = transitions.reward
        done = transitions.done

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

        vals = jnp.where(done, disc_returns_at_done, jnp.nan).reshape(-1)

        q1, q3 = jnp.nanquantile(vals, jnp.array([0.25, 0.75]))
        in_iqr = (~jnp.isnan(vals)) & (vals >= q1) & (vals <= q3)
        count = in_iqr.sum()
        total = jnp.where(in_iqr, vals, 0.0).sum()
        return jnp.where(count > 0, total / count, jnp.array(0.0, dtype=vals.dtype))

    @staticmethod
    @jax.jit
    def get_iqm_episodic_lengths(transitions):
        done = transitions.done

        def step(carry_len, done_t):
            curr_len = carry_len + 1
            out = jnp.where(done_t, curr_len, jnp.zeros_like(curr_len))
            next_len = jnp.where(done_t, jnp.zeros_like(curr_len), curr_len)
            return next_len, out

        init_len = jnp.zeros_like(done[0], dtype=jnp.int32)
        _, lengths_at_done = jax.lax.scan(step, init_len, done)

        vals = jnp.where(done, lengths_at_done, jnp.nan).astype(jnp.float32).reshape(-1)

        q1, q3 = jnp.nanquantile(vals, jnp.array([0.25, 0.75]))
        in_iqr = (~jnp.isnan(vals)) & (vals >= q1) & (vals <= q3)
        count = in_iqr.sum()
        total = jnp.where(in_iqr, vals, 0.0).sum()
        return jnp.where(count > 0, total / count, jnp.array(0.0, dtype=vals.dtype))

    @staticmethod
    @jax.jit
    def get_mmer_episodic_returns(transitions):
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
        return jnp.max(returns_at_done, initial=jnp.min(returns_at_done), where=done)

    @staticmethod
    @jax.jit
    def get_mmer_discounted_episodic_returns(transitions, gamma: float):
        r = transitions.reward
        done = transitions.done

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
        return jnp.max(
            disc_returns_at_done, initial=jnp.min(disc_returns_at_done), where=done
        )

    @staticmethod
    @jax.jit
    def get_mmer_episodic_lengths(transitions):
        done = transitions.done

        def step(carry_len, done_t):
            curr_len = carry_len + 1
            out = jnp.where(done_t, curr_len, jnp.zeros_like(curr_len))
            next_len = jnp.where(done_t, jnp.zeros_like(curr_len), curr_len)
            return next_len, out

        init_len = jnp.zeros_like(done[0], dtype=jnp.int32)
        _, lengths_at_done = jax.lax.scan(step, init_len, done)
        return jnp.max(lengths_at_done, initial=jnp.min(lengths_at_done), where=done)
