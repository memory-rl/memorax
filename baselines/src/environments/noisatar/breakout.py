import jax
import jax.numpy as jnp
from gymnax.environments.minatar.breakout import EnvState, MinBreakout


class POBreakout(MinBreakout):
    """A minimal partially observable implementation of the Atari Breakout environment for Gymnax."""

    def get_obs(self, state: EnvState, params=None, key=None) -> jax.Array:
        """Return observation from raw state trafo."""
        obs = jnp.zeros(self.obs_shape, dtype=jnp.bool)
        obs = obs.at[9, state.pos, 0].set(True)
        obs = obs.at[state.ball_y, state.ball_x, 1].set(True)
        # obs = obs.at[state.last_y, state.last_x, 2].set(True)
        obs = obs.at[:, :, 3].set(state.brick_map.astype(jnp.bool))
        mask = jax.random.bernoulli(key, p=0.1, shape=obs.shape[:2])
        mask = jnp.repeat(mask, obs.shape[2], axis=2)
        obs = jnp.where(mask, obs, jnp.zeros_like(obs))

        return obs.astype(jnp.float32)
