import jax
import jax.numpy as jnp
from gymnax.environments.minatar.asterix import EnvState, MinAsterix


class POAsterix(MinAsterix):
    """A minimal partially observable implementation of the Atari Asterix environment for Gymnax."""

    def get_obs(self, state: EnvState, params=None, key=None) -> jax.Array:
        """Return observation from raw state trafo."""
        # Add a 5th channel to help with not used entities
        obs = jnp.zeros((10, 10, 5), dtype=jnp.int32)
        # Set the position of the agent in the grid
        obs = obs.at[state.player_y, state.player_x, 0].set(1)
        # Loop over entity identities and set entity locations
        for i in range(state.entities.shape[0]):
            x = state.entities[i, :]
            # Enemy channel 1, Trail channel 2, Gold channel 3, Not used 4
            c = 3 * x[3] + 1 * (1 - x[3])
            c_eff = c * x[4] + 4 * (1 - x[4])
            obs = obs.at[x[1], x[0], c_eff].set(1)

            back_x = (x[0] - 1) * x[2] + (x[0] + 1) * (1 - x[2])
            leave_trail = jnp.logical_and(back_x >= 0, back_x <= 9)
            c_eff = 2 * x[4] + 4 * (1 - x[4])
            obs = obs.at[x[1], back_x, c_eff].set(leave_trail)

        px, py = state.player_x, state.player_y

        # NEW (Working)
        # 1. Create indices relative to the dynamic center
        y_idx = (py - 1) + jnp.arange(3)
        x_idx = (px - 1) + jnp.arange(3)
        c_idx = jnp.arange(1, obs.shape[-1])  # Channels usually static, so this is fine

        # 2. Use ix_ to create the index mesh
        indices = jnp.ix_(y_idx, x_idx, c_idx)

        # 3. Apply the update with mode='drop' to handle edges safely
        obs = obs.at[indices].set(0, mode="drop")

        return obs[:, :, :4].astype(jnp.float32)
