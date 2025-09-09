from typing import Optional
import chex


@chex.dataclass(frozen=True)
class Transition:
    obs: Optional[chex.Array] = None
    action: Optional[chex.Array] = None
    reward: Optional[chex.Array] = None
    done: Optional[chex.Array] = None
    info: Optional[dict] = None
    prev_done: Optional[chex.Array] = None
    next_obs: Optional[chex.Array] = None
    log_prob: Optional[chex.Array] = None
    value: Optional[chex.Array] = None
