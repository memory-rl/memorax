from dataclasses import field

import chex

from memory_rl.logger.logger import Logger


@chex.dataclass(frozen=True)
class ConsoleLogger(Logger):
    evaluation_metrics: list[str] = field(
        default_factory=lambda: [
            "evaluation/episodic_returns",
            "evaluation/episodic_lengths",
        ],
        hash=False,
    )
    training_metrics: list[str] = field(
        default_factory=lambda: [
            "training/episodic_returns",
            "training/episodic_lengths",
        ],
        hash=False,
    )

    def emit(self, data, step):
        metrics = " | ".join(
            [f"{k}: {v:.3f}" for k, v in data.items() if k in self.evaluation_metrics]
        )
        if self.debug:
            traininig_metrics = " | ".join(
                [f"{k}: {v:.3f}" for k, v in data.items() if k in self.training_metrics]
            )
            metrics = f"{metrics} | {traininig_metrics}"

        if metrics:
            print(f"Timestep: {step} - {metrics}")
