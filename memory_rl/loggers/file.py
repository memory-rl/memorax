from collections import defaultdict
from dataclasses import field
from datetime import datetime
from pathlib import Path
from typing import DefaultDict

from flax import struct

from .logger import BaseLogger, BaseLoggerState, PyTree


@struct.dataclass(frozen=True)
class FileLoggerState(BaseLoggerState):
    base: Path
    paths: dict[int, Path]
    buffer: DefaultDict[int, dict[str, PyTree]] = field(
        default_factory=lambda: defaultdict(dict)
    )


@struct.dataclass(frozen=True)
class FileLogger(BaseLogger[FileLoggerState]):
    algorithm: str
    environment: str
    directory: str = "logs"

    def init(self, cfg: dict) -> FileLoggerState:
        if "actor" in cfg["algorithm"]:
            cell = cfg["algorithm"]["actor"]["torso"]["_target_"]
        else:
            cell = cfg["algorithm"]["torso"]["_target_"]
        cell = cell.split(".")[-1]

        if "parameters" in cfg["environment"]:
            params = ""
            for key, param in cfg["environment"]["parameters"].items():
                if key == "max_steps_in_episode":
                    continue
                params += f"{param}/"

            if params:
                params = params[:-1]
            base_path = (
                Path(self.directory)
                / self.environment
                / params
                / self.algorithm
                / cell
                / f"{datetime.now():%Y%m%d-%H%M%S}"
            )
        else:
            base_path = (
                Path(self.directory)
                / self.environment
                / self.algorithm
                / cell
                / f"{datetime.now():%Y%m%d-%H%M%S}"
            )
        base_path.mkdir(exist_ok=True, parents=True)

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

                for metric, value in {
                    k: v[seed] if k != "SPS" else v for k, v in data.items()
                }.items():
                    metric_path = (path / f"{metric}.csv").resolve()
                    metric_path.parent.mkdir(exist_ok=True, parents=True)

                    if not metric_path.exists():
                        with metric_path.open("a") as f:
                            f.write(f"step,{metric}\n")

                    with metric_path.open("a") as f:
                        f.write(f"{step},{value}\n")

        state.buffer.clear()
        return state

    def finish(self, state: FileLoggerState) -> None:
        pass
