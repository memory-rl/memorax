from collections import defaultdict
from dataclasses import field
from typing import Optional

from flax import struct
from neptune_scale import Run

from .logger import BaseLogger, BaseLoggerState, PyTree


@struct.dataclass(frozen=True)
class NeptuneLoggerState(BaseLoggerState):
    runs: dict[int, Run]
    buffer: defaultdict[int, dict[str, PyTree]] = field(
        default_factory=lambda: defaultdict(dict)
    )


@struct.dataclass(frozen=True)
class NeptuneLogger(BaseLogger[NeptuneLoggerState]):
    workspace: Optional[str] = None
    project: Optional[str] = None
    mode: str = "disabled"

    def init(self, cfg: dict) -> NeptuneLoggerState:
        runs = {
            seed: Run(project=f"{self.workspace}/{self.project}", mode=self.mode)
            for seed in range(cfg["num_seeds"])
        }
        for run in runs.values():
            run.log_configs(cfg)

        return NeptuneLoggerState(runs=runs)

    def log(
        self, state: NeptuneLoggerState, data: PyTree, step: int
    ) -> NeptuneLoggerState:
        state.buffer[step].update(data)
        return state

    def emit(self, state: NeptuneLoggerState) -> NeptuneLoggerState:
        for step, data in sorted(state.buffer.items()):
            for seed, run in state.runs.items():
                run.log_metrics(
                    {k: v[seed] if k != "SPS" else v for k, v in data.items()},
                    step=step,
                )
        state.buffer.clear()
        return state

    def finish(self, state: NeptuneLoggerState) -> None:
        for run in state.runs.values():
            run.close()
