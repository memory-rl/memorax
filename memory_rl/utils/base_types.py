import dataclasses
from typing import Any, Dict, Optional

import chex
from flashbax.buffers.trajectory_buffer import BufferState
from flax.core.frozen_dict import FrozenDict
from flax.training import train_state
from jumanji.types import TimeStep
from typing_extensions import TypeAlias

# Can't know the exact type of State.
State: TypeAlias = Any
TrainState: TypeAlias = Any
EnvState: TypeAlias = Any


@chex.dataclass(frozen=True)
class OnPolicyLearnerState:
    """Stores all the necessary states for on-policy learners."""

    key: chex.PRNGKey
    env_state: EnvState
    step: int
    ### Optional fields because gymnax doesn't store these ###
    ### Constructor is screaming because later args will follow optional ones ###
    # obs: Optional[chex.Array] = None
    # done: Optional[chex.Array] = None
    ##########################################################


@chex.dataclass(frozen=True)
class OffPolicyLearnerState(OnPolicyLearnerState):
    """Augment on-policy learner state with buffer state."""

    buffer_state: BufferState


@chex.dataclass(frozen=True)
class RNNOffPolicyLearnerState(OffPolicyLearnerState):
    """Augment off-policy learner state with RNN hidden state
    necessary for online inference."""

    hidden_state: tuple


class OnlineAndTargetState(train_state.TrainState):
    """Augmented train state with target parameters."""

    target_params: FrozenDict

    @classmethod
    def create(cls, *, apply_fn, params, tx, opt_state, target_params, **kwargs):
        """Creates a new instance with `step=0`, initialized `opt_state`, and `target_params`."""

        return cls(
            step=0,
            apply_fn=apply_fn,
            params=params,
            tx=tx,
            opt_state=opt_state,
            target_params=target_params,
            **kwargs
        )


### Illustrative examples

# class SACTrainState(NamedTuple):
#     actor: train_state.TrainState
#     q: OnlineAndTargetState
#     alpha: train_state.TrainState


# class DRQNTrainState(NamedTuple):
#     q: OnlineAndTargetState
