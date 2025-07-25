import chex
from omegaconf import DictConfig


@chex.dataclass(frozen=True)
class Logger:
    log_interval: int = (
        1  # be careful with this, as it should be a multiple of the number of envs
    )
    debug: bool = False

    def init(self, cfg: DictConfig):
        pass

    def log(self, data, step):
        if step % self.log_interval == 0:
            self.emit(data, step)

    def emit(self, data, step):
        pass
