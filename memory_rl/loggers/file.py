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
    paths: dict[int, Path]
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

        base_path = (
            Path(self.directory)
            / self.environment
            / self.algorithm
            / cell
            / f"{datetime.now():%Y%m%d-%H%M%S}"
        )
        base_path.mkdir(exist_ok=True, parents=True)
        OmegaConf.save(OmegaConf.create(cfg), base_path / "config.yaml")

        paths = {seed: (base_path / str(seed)) for seed in range(cfg["num_seeds"])}

        for _, path in paths.items():
            path.mkdir(exist_ok=True, parents=True)

        return FileLoggerState(base=base_path, paths=paths)

    def log(self, state: FileLoggerState, data: PyTree, step: int) -> FileLoggerState:
        state.buffer[step].update(data)
        return state

    def emit(self, state: FileLoggerState) -> FileLoggerState:
        for step, data in sorted(state.buffer.items()):
            for seed, path in state.paths.items():

                for metric, value in {k: v[seed] if k != "SPS" else v for k, v in data.items()}.items():
                    metric_path = (path / f"{metric}.csv").resolve()
                    metric_path.parent.mkdir(exist_ok=True, parents=True)

                    with metric_path.open("a") as f:
                        f.write(f"{step},{value}\n")

        state.buffer.clear()
        return state

    def finish(self, state: FileLoggerState) -> None:
        pass
