from flax import struct

from memorax.utils.typing import Array


@struct.dataclass(frozen=True)
class Timestep:
    obs: Array
    action: Array
    reward: Array
    done: Array
