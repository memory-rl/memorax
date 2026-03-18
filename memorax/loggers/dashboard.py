from collections import defaultdict
from typing import Any

import jax
import jax.numpy as jnp
from rich import box
from rich.console import Console
from rich.live import Live
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table

from memorax.utils.typing import PyTree


class DashboardLogger:
    def __init__(self, total_timesteps=0, refresh_per_second=10, summary=None, **kwargs):
        self.summary = summary or {}

        self.console = Console()

        self.progress = Progress(
            TextColumn("[progress.description]{task.description}"),
            SpinnerColumn(),
            TimeElapsedColumn(),
            BarColumn(bar_width=None),
            TimeRemainingColumn(),
            expand=True,
            console=self.console,
        )
        self.progress_task = self.progress.add_task("Progress", total=total_timesteps)

        dashboard = self.build_dashboard({}, 0, self.progress, self.progress_task)

        self.live = Live(
            dashboard,
            console=self.console,
            refresh_per_second=refresh_per_second,
            transient=False,
        )
        self.live.start()

    def log(self, data: PyTree, step: int, **kwargs):
        self.progress.update(self.progress_task, completed=int(step))
        dashboard = self.build_dashboard(data, step, self.progress, self.progress_task)
        self.live.update(dashboard, refresh=True)

    def finish(self) -> None:
        self.live.stop()
        self.console.show_cursor(True)

    def group(self, data: dict[str, PyTree]) -> dict[str, dict[str, Any]]:
        data = {
            "/".join(str(p.key) for p in path): leaf
            for path, leaf in jax.tree_util.tree_leaves_with_path(data)
        }
        groups = defaultdict(dict)
        for key, value in data.items():
            if "/" in key:
                prefix, name = key.split("/", 1)
                groups[prefix][name] = value
            else:
                groups[""][key] = value
        return dict(groups)

    def build_table(self, heading: str, metrics: dict[str, PyTree]) -> Table:
        table = Table(box=None, expand=True)
        table.add_column(heading, justify="left", width=20, style="yellow")
        table.add_column("Value", justify="right", width=10, style="green")
        for name, value in metrics.items():
            table.add_row(name, f"{jnp.mean(value):.3f}")
        return table

    def build_dashboard(
        self, data: dict[str, PyTree], step: int, progress: Progress, task: Any
    ) -> Table:
        dashboard = Table(
            box=box.ROUNDED,
            expand=True,
            show_header=False,
            border_style="white",
        )

        dynamic_summary = {
            k.split("/", 1)[1]: v for k, v in data.items() if k.startswith("summary/")
        }
        items = [*self.summary.items(), *dynamic_summary.items()]
        if data:
            items.append(("Step", f"{int(step):_}"))
        left = Table(box=None, expand=True)
        left.add_column("Summary", justify="left", width=16, style="white")
        left.add_column("Value", justify="right", width=8, style="white")
        right = Table(box=None, expand=True)
        right.add_column("Summary", justify="left", width=16, style="white")
        right.add_column("Value", justify="right", width=8, style="white")
        for i, (key, value) in enumerate(items):
            table = left if i % 2 == 0 else right
            table.add_row(key, f"{value}", style="white")
        summary_row = Table(box=None, expand=True, pad_edge=False)
        summary_row.add_row(left, right)
        dashboard.add_row(summary_row)

        groups = self.group(data)
        groups.pop("summary", None)
        group_names = list(groups.keys())

        for i in range(0, len(group_names), 2):
            pair = group_names[i : i + 2]
            tables = [self.build_table(name, groups[name]) for name in pair]
            row = Table(box=None, expand=True, pad_edge=False)
            row.add_row(*tables)
            dashboard.add_row(row)

        dashboard.add_row("")
        progress.update(task, completed=int(step))
        dashboard.add_row(progress)

        return dashboard
