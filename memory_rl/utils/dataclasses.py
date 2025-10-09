from typing import Optional
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
