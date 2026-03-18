from pathlib import Path

from memorax.utils.typing import PyTree


class FileLogger:
    def __init__(self, directory="logs", **kwargs):
        self.directory = Path(directory)
        self.directory.mkdir(exist_ok=True, parents=True)

    def log(self, data: PyTree, step: int, **kwargs):
        for metric, value in data.items():
            path = (self.directory / f"{metric}.csv").resolve()
            path.parent.mkdir(exist_ok=True, parents=True)
            header = not path.exists()
            with open(path, "a") as file:
                if header:
                    file.write(f"step,{metric}\n")
                file.write(f"{step},{value}\n")

    def finish(self) -> None:
        pass
