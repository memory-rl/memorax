import jax
from tensorboardX import SummaryWriter

from memorax.utils.axes import ensure_axis
from memorax.utils.typing import PyTree


class TensorBoardLogger:
    def __init__(self, directory="tensorboard", num_seeds=1, **kwargs):
        self.writers = {
            seed: SummaryWriter(log_dir=directory) for seed in range(num_seeds)
        }

    def log(self, data: PyTree, step: int, **kwargs):
        num_seeds = len(self.writers)
        data = jax.tree.map(lambda v: ensure_axis(v, num_seeds), data)
        for seed, writer in self.writers.items():
            for metric, value in data.items():
                writer.add_scalar(metric, value[seed], step)

    def finish(self) -> None:
        for writer in self.writers.values():
            writer.close()
