from typing import Optional

import jax
from flax import struct

from memorax.networks.sequence_models.utils import add_time_axis, remove_time_axis
from memorax.utils.typing import Array


@struct.dataclass(frozen=True)
class Timestep:
    obs: Optional[Array] = None
    action: Optional[Array] = None
    reward: Optional[Array] = None
    done: Optional[Array] = None

    def to_sequence(self):
        return jax.tree.map(add_time_axis, self)

    def from_sequence(self):
        return jax.tree.map(remove_time_axis, self)
