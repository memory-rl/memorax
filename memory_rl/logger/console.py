from dataclasses import dataclass, field

from memory_rl.logger.logger import Logger


@dataclass
class ConsoleLogger(Logger):
    metrics: list[str] = field(
        default_factory=lambda: ["episodic_returns", "episodic_lengths"]
    )

    def log(self, data, step):
        metrics = " | ".join(
            [f"{k}: {v:.3f}" for k, v in data.items() if k in self.metrics]
        )
        print(f"Timestep: {step} - {metrics}")
