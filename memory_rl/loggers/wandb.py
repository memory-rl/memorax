from collections import defaultdict
from dataclasses import field
from .logger import BaseLogger, BaseLoggerState, PyTree

import chex
from typing import Any, Optional
import wandb
from wandb.sdk.wandb_run import Run


@chex.dataclass(frozen=True)
class WandbLoggerState(BaseLoggerState):
    runs: dict[int, Run]
    buffer: defaultdict[int, dict[str, PyTree]] = field(
        default_factory=lambda: defaultdict(dict)
    )


@chex.dataclass(frozen=True)
class WandbLogger(BaseLogger[WandbLoggerState]):
    entity: Optional[str] = None
    project: Optional[str] = None
    name: Optional[str] = None
    group: Optional[str] = None
    mode: str = "disabled"

    def init(self, cfg: dict) -> WandbLoggerState:
        runs = {
            seed: wandb.init(
                entity=self.entity,
                project=self.project,
                name=self.name,
                group=self.group,
                mode=self.mode,
                config=cfg,
                reinit="create_new",
            )
            for seed in range(cfg["num_seeds"])
        }
        return WandbLoggerState(runs=runs)

    def log(self, state: WandbLoggerState, data: PyTree, step: int) -> WandbLoggerState:
        state.buffer[step].update(data)
        return state

    def emit(self, state: WandbLoggerState) -> WandbLoggerState:
        for step, data in sorted(state.buffer.items()):
            for seed, run in state.runs.items():
                run.log(
                    {k: v[seed] if k != "SPS" else v for k, v in data.items()},
                    step=step,
                )

        state.buffer.clear()
        return state

    def finish(self, state: WandbLoggerState) -> None:
        for run in state.runs.values():
            run.finish()
