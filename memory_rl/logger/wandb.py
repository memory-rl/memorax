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

    def init(self, cfg: DictConfig):
        wandb.init(
            entity=self.entity,
            project=self.project,
            name=self.name,
            group=self.group,
            mode=self.mode,
            config=cfg,
        )

    def emit(self, data, step):
        super(WandbLogger, self).emit(
            {k: v for k, v in data.items() if k.startswith("evaluation")}, step
        )
        wandb.log(data, step=step)
