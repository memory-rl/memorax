import jax
import jax.numpy as jnp
from gymnax.environments.minatar.freeway import EnvState, MinFreeway


class POFreeway(MinFreeway):
    """A minimal partially observable implementation of the Atari Freeway environment for Gymnax."""

    def get_obs(self, state: EnvState, params=None, key=None) -> jax.Array:
        """Return observation from raw state trafo."""
        obs = jnp.zeros(self.obs_shape, dtype=bool)
        # Set the position of the chicken agent, cars, and trails
        obs = obs.at[state.pos, 4, 0].set(True)
        for car_id in range(8):
            car = state.cars[car_id]
            obs = obs.at[car[1], car[0], 1].set(True)
            # Boundary conditions for cars
            back_x = (car[3] > 0).astype(jnp.int32) * (car[0] - 1) + (
                1 - (car[3] > 0).astype(jnp.int32)
            ) * (car[0] + 1)
            left_out = (back_x < 0).astype(jnp.int32)
            right_out = (back_x > 9).astype(jnp.int32)
            back_x = left_out * 9 + (1 - left_out) * back_x
            back_x = right_out * 0 + (1 - right_out) * back_x
            # Set trail to be on
            trail_channel = (
                2 * (jnp.abs(car[3]) == 1).astype(jnp.int32)
                + 3 * (jnp.abs(car[3]) == 2).astype(jnp.int32)
                + 4 * (jnp.abs(car[3]) == 3).astype(jnp.int32)
                + 5 * (jnp.abs(car[3]) == 4).astype(jnp.int32)
                + 6 * (jnp.abs(car[3]) == 5).astype(jnp.int32)
            )
            obs = obs.at[car[1], back_x, trail_channel].set(True)
        obs = obs.at[:, 2:-2, 1:].set(False)  # Remove empty space
        return obs.astype(jnp.float32)
