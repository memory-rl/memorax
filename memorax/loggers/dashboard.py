from collections import defaultdict
from dataclasses import field
from typing import Any, DefaultDict, Optional

from flax import struct
from rich import box
from rich.console import Console
from rich.live import Live
from rich.progress import BarColumn, Progress, TextColumn, TimeRemainingColumn
from rich.table import Table

from .logger import BaseLogger, BaseLoggerState, PyTree


@struct.dataclass(frozen=True)
class DashboardLoggerState(BaseLoggerState):
    console: Console
    live: Live
    progress: Progress
    progress_task: Any
    buffer: DefaultDict[int, dict[str, PyTree]] = field(
        default_factory=lambda: defaultdict(dict)
    )
    stats: dict[str, Any] = field(
        default_factory=lambda: {
            "global_step": 0,
            "training/SPS": 0,
            "evaluation/SPS": 0,
            "losses": {},  # dict[str, float]
            "metrics": {},  # dict[str, float]
        }
    )


@struct.dataclass(frozen=True)
class DashboardLogger(BaseLogger[DashboardLoggerState]):
    title: Optional[str]
    name: Optional[str] = None
    total_timesteps: int = 0
    refresh_per_second: int = 10
    env_id: Optional[str] = None

    def init(self, **kwargs) -> DashboardLoggerState:
        console = Console()

        progress = Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=None),
            TimeRemainingColumn(),
            expand=True,
            console=console,
        )
        task = progress.add_task("Progress", total=self.total_timesteps)

        dashboard = self.get_dashboard(
            stats={
                "global_step": 0,
                "training/SPS": 0,
                "evaluation/SPS": 0,
                "losses": {},
                "metrics": {},
            },
            progress=progress,
            task=task,
        )

        live = Live(
            dashboard,
            console=console,
            refresh_per_second=self.refresh_per_second,
            transient=False,
        )
        live.start()

        return DashboardLoggerState(
            console=console,
            live=live,
            progress=progress,
            progress_task=task,
        )

    def log(
        self, state: DashboardLoggerState, data: PyTree, step: int
    ) -> DashboardLoggerState:
        state.buffer[step].update(data)
        return state

    def emit(self, state: DashboardLoggerState) -> DashboardLoggerState:
        for step, data in sorted(state.buffer.items()):
            state.stats["global_step"] = max(state.stats["global_step"], step)

            state.stats["training/SPS"] = data.pop(
                "training/SPS", state.stats["training/SPS"]
            )
            state.stats["evaluation/SPS"] = data.pop(
                "evaluation/SPS", state.stats["evaluation/SPS"]
            )

            state.stats["losses"].update(
                {k: v.mean() for k, v in data.items() if k.startswith("losses/")}
            )
            state.stats["metrics"].update(
                {
                    k: v.mean()
                    for k, v in data.items()
                    if k.startswith("training/") or k.startswith("evaluation/")
                }
            )

        state.buffer.clear()

        state.progress.update(
            state.progress_task, completed=int(state.stats["global_step"])
        )
        dashboard = self.get_dashboard(state.stats, state.progress, state.progress_task)
        state.live.update(dashboard, refresh=True)
        return state

    def finish(self, state: DashboardLoggerState) -> None:
        state.progress.update(
            state.progress_task, completed=int(state.stats["global_step"])
        )
        state.live.update(
            self.get_dashboard(state.stats, state.progress, state.progress_task),
            refresh=True,
        )
        state.live.stop()

    def get_dashboard(
        self, stats: dict[str, Any], progress: Progress, task: Any
    ) -> Table:
        dashboard = Table(
            box=box.ROUNDED,
            expand=True,
            show_header=False,
            border_style="white",
        )

        header = Table(box=None, expand=True, show_header=False)
        header.add_column(justify="left")
        header.add_row(f"[bold white]{self.title} - {self.name}[/]")
        dashboard.add_row(header)

        summary_table = Table(box=None, expand=True)
        summary_table.add_column(
            "Summary", justify="left", vertical="top", width=16, style="white"
        )
        summary_table.add_column(
            "Value", justify="right", vertical="top", width=8, style="white"
        )
        summary_table.add_row("Environment", f"{self.env_id}", style="white")
        summary_table.add_row(
            "Total Timesteps", f"{self.total_timesteps:_}", style="white"
        )
        summary_table.add_row(
            "Global Step", f"{int(stats['global_step']):_}", style="white"
        )
        summary_table.add_row(
            "training/SPS", f"{int(stats['training/SPS']):_}", style="white"
        )
        summary_table.add_row(
            "evaluation/SPS", f"{int(stats['evaluation/SPS']):_}", style="white"
        )

        losses_table = Table(box=None, expand=True)
        losses_table.add_column("Losses", justify="left", width=16, style="white")
        losses_table.add_column("Value", justify="right", width=8, style="white")
        for metric, value in stats["losses"].items():
            losses_table.add_row(str(metric), f"{value:.{3}f}")

        monitor = Table(box=None, expand=True, pad_edge=False)
        monitor.add_row(summary_table, losses_table)
        dashboard.add_row(monitor)

        statistics = Table(box=None, expand=True, pad_edge=False)
        left_stats = Table(box=None, expand=True)
        right_stats = Table(box=None, expand=True)
        left_stats.add_column("Training", justify="left", width=20, style="yellow")
        left_stats.add_column("Value", justify="right", width=10, style="green")
        right_stats.add_column("Evaluation", justify="left", width=20, style="yellow")
        right_stats.add_column("Value", justify="right", width=10, style="green")
        for i, (metric, value) in enumerate(stats["metrics"].items()):
            if metric.startswith("training/"):
                table = left_stats
            elif metric.startswith("evaluation/"):
                table = right_stats
            else:
                print(f"Unknown metric: {metric}")
                continue

            name = metric.split("/")[-1]
            table.add_row(name, f"{value:.{3}f}")

        statistics.add_row(left_stats, right_stats)
        dashboard.add_row(statistics)

        dashboard.add_row("")
        progress.update(task, completed=int(stats["global_step"]))
        dashboard.add_row(progress)

        return dashboard
