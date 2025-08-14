import chex
from omegaconf import DictConfig

import wandb
from memory_rl.logger.console import ConsoleLogger

@chex.dataclass(frozen=True)
class WandbLogger(ConsoleLogger):

    entity: str | None = None
    project: str | None = None
    name: str | None = None
    group: str | None = None
    mode: str = "disabled"
    run: wandb.sdk.wandb_run.Run | None = None

    def init(self, cfg: DictConfig):
        run = wandb.init(
            entity=self.entity,
            project=self.project,
            name=self.name,
            group=self.group,
            mode=self.mode,
            config=cfg,
            reinit="create_new"
        )
        object.__setattr__(self, "run", run)

    def emit(self, data, step):
        super(WandbLogger, self).emit(
            data, step=step
        )
        assert self.run is not None, "WandB run not initialized"
        self.run.log(data, step=step)

    def finish(self):
        if self.run is not None:
            self.run.finish()
