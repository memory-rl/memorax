from typing import Optional
import chex


@chex.dataclass(frozen=True)
class Transition:
    obs: chex.Array
    done: chex.Array
    action: chex.Array
    reward: chex.Array
    next_obs: Optional[chex.Array] = None
    next_done: Optional[chex.Array] = None
    info: Optional[dict] = None
    log_prob: Optional[chex.Array] = None
    value: Optional[chex.Array] = None
