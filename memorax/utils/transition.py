from typing import Optional

import jax
import jax.numpy as jnp
from flax import struct

from memorax.utils.typing import Array, PyTree


@struct.dataclass(frozen=True)
class Transition:
    obs: Optional[PyTree] = None
    action: Optional[PyTree] = None
    reward: Optional[PyTree] = None
    done: Optional[PyTree] = None
    info: Optional[dict] = None
    prev_action: Optional[PyTree] = None
    prev_reward: Optional[PyTree] = None
    prev_done: Optional[PyTree] = None
    next_obs: Optional[PyTree] = None
    log_prob: Optional[PyTree] = None
    value: Optional[PyTree] = None
    env_state: Optional[PyTree] = None
    carry: Optional[PyTree] = None

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

    @property
    def infos(self):
        assert self.info is not None
        return jax.tree.map(
            jnp.mean,
            {k: v for k, v in self.info.items() if not k.startswith("losses")},
        )
