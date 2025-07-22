from dataclasses import field

import chex

from memory_rl.logger.logger import Logger


@chex.dataclass(frozen=True)
class ConsoleLogger(Logger):
    metrics: list[str] = field(
        default_factory=lambda: [
            "evaluation/episodic_returns",
            "evaluation/episodic_lengths",
            "training/episodic_returns",
            "training/episodic_lengths",
        ],
        hash=False,
    )

    def emit(self, data, step):
        metrics = " | ".join(
            [f"{k}: {v:.3f}" for k, v in data.items() if k in self.metrics]
        )
        if metrics:
            print(f"Timestep: {step} - {metrics}")
