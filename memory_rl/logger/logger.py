from dataclasses import dataclass
from omegaconf import DictConfig


@dataclass
class Logger:

    def init(self, cfg: DictConfig):
        pass

    def log(self, data, step):
        pass
