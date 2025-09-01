from __future__ import annotations

from pathlib import Path
from collections import defaultdict
from dataclasses import field
from typing import Any, DefaultDict, Mapping, Optional
from datetime import datetime

import numpy as np
import jax
import chex
from omegaconf import DictConfig, OmegaConf

from .logger import BaseLogger, BaseLoggerState, PyTree


@chex.dataclass(frozen=True)
class FileLoggerState(BaseLoggerState):
    base: Path
    buffer: DefaultDict[int, dict[str, PyTree]] = field(
        default_factory=lambda: defaultdict(dict)
    )


@chex.dataclass(frozen=True)
class FileLogger(BaseLogger[FileLoggerState]):
    algorithm: str
    environment: str
    seed: int
    directory: str = "logs"

    def init(self, cfg: dict) -> FileLoggerState:
        cell = cfg["algorithm"]["actor"]["torso"]["_target_"].split(".")[-1]
        path = (
            Path(self.directory)
            / self.algorithm
            / self.environment
            / cell
            / str(self.seed)
            / f"{datetime.now():%Y%m%d-%H%M%S}"
        )
        path.mkdir(exist_ok=True, parents=True)

        OmegaConf.save(OmegaConf.create(cfg), path / "config.yaml")

        return FileLoggerState(base=path)

    def log(self, state: FileLoggerState, data: PyTree, step: int) -> FileLoggerState:
        state.buffer[step].update(data)
        return state

    def emit(self, state: FileLoggerState) -> FileLoggerState:
        for step, data in sorted(state.buffer.items()):
            for metric, value in data.items():
                path = (state.base / f"{metric}.csv").resolve()
                path.parent.mkdir(exist_ok=True, parents=True)

                with path.open("a") as f:
                    f.write(f"{step},{value}\n")

        state.buffer.clear()
        return state

    def finish(self, state: FileLoggerState) -> None:
        pass
