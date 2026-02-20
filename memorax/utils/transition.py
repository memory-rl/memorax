from typing import Optional

import jax
import jax.numpy as jnp
from flax import struct

from memorax.utils.timestep import Timestep
from memorax.utils.typing import Array, PyTree


@struct.dataclass(frozen=True)
class Transition:
    first: Optional[Timestep] = None
    second: Optional[Timestep] = None
    log_prob: Optional[PyTree] = None
    value: Optional[PyTree] = None
    carry: Optional[PyTree] = None
    metadata: Optional[dict] = None

    @property
    def num_episodes(self) -> Array:
        assert self.second is not None and self.second.done is not None
        return self.second.done.sum()

    @property
    def episode_lengths(self):
        assert self.second is not None and self.second.done is not None
        done = self.second.done

        def step(carry_len, done_t):
            curr_len = carry_len + 1
            out = jnp.where(done_t, curr_len, jnp.zeros_like(curr_len))
            next_len = jnp.where(done_t, jnp.zeros_like(curr_len), curr_len)
            return next_len, out

        init_len = jnp.zeros_like(done[0], dtype=jnp.int32)
        _, episode_lengths = jax.lax.scan(step, init_len, done)
        return jnp.where(done, episode_lengths, jnp.nan)

    @property
    def episode_returns(self):
        assert self.second is not None
        assert self.second.reward is not None and self.second.done is not None
        reward = self.second.reward
        done = self.second.done

        def step(carry_sum, inp):
            r_t, d_t = inp
            s = carry_sum + r_t
            out = jnp.where(d_t, s, jnp.zeros_like(s))
            next_s = jnp.where(d_t, jnp.zeros_like(s), s)
            return next_s, out

        init_sum = jnp.zeros_like(reward[0])
        _, episode_returns = jax.lax.scan(step, init_sum, (reward, done))
        return jnp.where(done, episode_returns, jnp.nan)

    @property
    def losses(self):
        if self.metadata is None:
            return {}
        return {k: v.mean() for k, v in self.metadata.items() if k.startswith("losses")}

    @property
    def info(self):
        if self.metadata is None:
            return {}
        return jax.tree.map(
            jnp.mean,
            {k: v for k, v in self.metadata.items() if not k.startswith("losses")},
        )
