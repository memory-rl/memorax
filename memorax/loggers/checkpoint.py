import jax
import numpy as np
import orbax.checkpoint.experimental.v1 as ocp

from memorax.utils.decorators import callback


class CheckpointLogger:
    def __init__(self, directory="checkpoints", max_to_keep=None, **kwargs):
        preservation_policy = None
        if max_to_keep is not None:
            preservation_policy = ocp.training.preservation_policies.LatestN(
                max_to_keep
            )
        self.checkpointer = ocp.training.Checkpointer(
            directory,
            preservation_policy=preservation_policy,
        )

    @callback
    def log(self, data, step, train_state=None, **kwargs):
        if train_state is None:
            return
        train_state = jax.tree.map(lambda value: np.asarray(value), train_state)
        self.checkpointer.save_pytree(int(step), train_state, force=True)

    def finish(self):
        self.checkpointer.close()
