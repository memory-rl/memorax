from typing import Optional

import jax
import jax.numpy as jnp
from flax import struct

from memory_rl.utils.typing import Array


@struct.dataclass(frozen=True)
class Transition:
    obs: Optional[Array] = None
    action: Optional[Array] = None
    reward: Optional[Array] = None
    done: Optional[Array] = None
    info: Optional[dict] = None
    prev_done: Optional[Array] = None
    next_obs: Optional[Array] = None
    log_prob: Optional[Array] = None
    value: Optional[Array] = None
    env_state: Optional[Array] = None

    @property
    def num_episodes(self) -> Array:
        assert self.done is not None
        return self.done.sum()

    @property
    def episode_lengths(self):
        assert self.done is not None

        def step(carry_len, done_t):
            curr_len = carry_len + 1
            out = jnp.where(done_t, curr_len, jnp.zeros_like(curr_len))
            next_len = jnp.where(done_t, jnp.zeros_like(curr_len), curr_len)
            return next_len, out

        init_len = jnp.zeros_like(self.done[0], dtype=jnp.int32)
        _, episode_lengths = jax.lax.scan(step, init_len, self.done)
        return jnp.where(self.done, episode_lengths, jnp.nan)

    @property
    def episode_returns(self):
        assert self.reward is not None
        assert self.done is not None

        def step(carry_sum, inp):
            r_t, d_t = inp
            s = carry_sum + r_t
            out = jnp.where(d_t, s, jnp.zeros_like(s))
            next_s = jnp.where(d_t, jnp.zeros_like(s), s)
            return next_s, out

        init_sum = jnp.zeros_like(self.reward[0])
        _, episode_returns = jax.lax.scan(step, init_sum, (self.reward, self.done))
        return jnp.where(self.done, episode_returns, jnp.nan)

    @property
    def losses(self):
        assert self.info is not None
        return {k: v.mean() for k, v in self.info.items() if k.startswith("losses")}
