import jax
import jax.numpy as jnp
from flax import struct

from memorax.utils.typing import Array
from memorax.networks.sequence_models.utils import add_time_axis, remove_time_axis


@struct.dataclass(frozen=True)
class Timestep:
    obs: Array
    action: Array
    reward: Array
    done: Array

    def to_sequence(self):
        return jax.tree.map(add_time_axis, self)

    def from_sequence(self):
        return jax.tree.map(remove_time_axis, self)
