from collections import defaultdict
from dataclasses import field

from flax import struct

from .logger import BaseLogger, BaseLoggerState, PyTree

from tensorboardX import SummaryWriter


@struct.dataclass(frozen=True)
class TensorBoardLoggerState(BaseLoggerState):
    writers: dict[int, SummaryWriter]
    buffer: defaultdict[int, dict[str, PyTree]] = field(
        default_factory=lambda: defaultdict(dict)
    )


@struct.dataclass(frozen=True)
class TensorBoardLogger(BaseLogger[TensorBoardLoggerState]):
    log_dir: str = "tensorboard"

    def init(self, cfg: dict) -> TensorBoardLoggerState:
        writers = {
            seed: SummaryWriter(log_dir=self.log_dir)
            for seed in range(cfg["num_seeds"])
        }
        return TensorBoardLoggerState(writers=writers)

    def log(
        self, state: TensorBoardLoggerState, data: PyTree, step: int
    ) -> TensorBoardLoggerState:
        state.buffer[step].update(data)
        return state

    def emit(self, state: TensorBoardLoggerState) -> TensorBoardLoggerState:
        for step, data in sorted(state.buffer.items()):
            for seed, writer in state.writers.items():
                for metric, value in data.items():
                    writer.add_scalar(
                        metric,
                        value[seed] if metric != "SPS" else value,
                        step,
                    )
        state.buffer.clear()
        return state

    def finish(self, state: TensorBoardLoggerState) -> None:
        for writer in state.writers.values():
            writer.close()
