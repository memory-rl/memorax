import jax
import jax.numpy as jnp
from gymnax.environments.minatar.space_invaders import EnvState, MinSpaceInvaders


class POSpaceInvaders(MinSpaceInvaders):
    """A minimal partially observable implementation of the Atari SpaceInvaders environment for Gymnax."""

    def get_obs(self, state: EnvState, params=None, key=None) -> jax.Array:
        """Return observation from raw state trafo."""
        obs = jnp.zeros((10, 10, 6), dtype=bool)
        # Update cannon, aliens - left + right dir, friendly + enemy bullet
        obs = obs.at[9, state.pos, 0].set(True)
        obs = obs.at[:, :, 1].set(state.alien_map.astype(jnp.bool))
        # left_dir_cond = state.alien_dir < 0
        # obs = jax.lax.select(
        #     left_dir_cond,
        #     obs.at[:, :, 2].set(state.alien_map.astype(jnp.bool)),
        #     obs.at[:, :, 2].set(state.alien_map.astype(jnp.bool)),
        #     obs.at[:, :, 3].set(state.alien_map.astype(jnp.bool)),
        # )
        bullet_map = jnp.logical_or(state.f_bullet_map, state.e_bullet_map)
        obs = obs.at[:, :, 4].set(bullet_map.astype(jnp.bool))
        obs = obs.at[:, :, 5].set(bullet_map.astype(jnp.bool))
        return obs.astype(jnp.float32)
