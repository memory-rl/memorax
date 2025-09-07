#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import argparse, os, math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Set
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------- I/O ----------


def try_read_metric_csv(path: Path) -> Optional[pd.DataFrame]:
    try:
        df = read_metric_csv(path)
        if (
            len(df) >= 2
            and pd.to_numeric(df["step"], errors="coerce").notna().sum() >= 2
        ):
            return df
    except Exception:
        pass
    return None


def read_metric_csv(path: Path) -> pd.DataFrame:
    """Return DataFrame ['step','value']; handles headerless or headered CSVs; skips folders."""
    if not path.is_file():
        raise FileNotFoundError(path)
    # Try headerless first
    try:
        df0 = pd.read_csv(path, header=None)
        if df0.shape[1] >= 2:
            s = pd.to_numeric(df0.iloc[:, 0], errors="coerce")
            v = pd.to_numeric(df0.iloc[:, 1], errors="coerce")
            ok = s.notna() & v.notna()
            if ok.sum() >= max(2, int(0.8 * len(df0))):
                out = pd.DataFrame({"step": s[ok], "value": v[ok]})
                return (
                    out.sort_values("step")
                    .drop_duplicates("step")
                    .reset_index(drop=True)
                )
    except Exception:
        pass
    # Fallback: headered
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    step_col = next(
        (
            cols[k]
            for k in [
                "step",
                "steps",
                "timestep",
                "timesteps",
                "env_step",
                "env_steps",
                "global_step",
            ]
            if k in cols
        ),
        None,
    )
    if step_col is None:
        numeric = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if not numeric:
            raise ValueError(f"No numeric step column in {path}")
        step_col = numeric[0]
    value_col = next(
        (cols[k] for k in ["value", "values", "metric", "y"] if k in cols), None
    )
    if value_col is None:
        numeric = [
            c
            for c in df.columns
            if c != step_col and pd.api.types.is_numeric_dtype(df[c])
        ]
        if not numeric:
            raise ValueError(f"No numeric value column in {path}")
        value_col = numeric[0]
    out = pd.DataFrame(
        {
            "step": pd.to_numeric(df[step_col], errors="coerce"),
            "value": pd.to_numeric(df[value_col], errors="coerce"),
        }
    ).dropna()
    return out.sort_values("step").drop_duplicates("step").reset_index(drop=True)


def pick_latest_timestamp_dir(seed_dir: Path) -> Optional[Path]:
    if not seed_dir.exists():
        return None
    ts = [p for p in seed_dir.iterdir() if p.is_dir()]
    return sorted(ts)[-1] if ts else None


def looks_like_algo_dir(d: Path) -> bool:
    """
    Heuristic: an algo dir has child 'cells' dirs, which contain 'seed' dirs.
    This is lightweight and works for distinguishing setting/variant dirs from algos.
    """
    try:
        for cell_dir in d.iterdir():
            if not cell_dir.is_dir():
                continue
            for seed_dir in cell_dir.iterdir():
                if seed_dir.is_dir():
                    return True
    except Exception:
        pass
    return False


def discover_algos(base: Path) -> List[Path]:
    """Algo dirs directly under 'base'."""
    return sorted([p for p in base.iterdir() if p.is_dir() and looks_like_algo_dir(p)])


def discover_runs_for_algo(
    algo_dir: Path, cells: Optional[List[str]]
) -> Dict[str, List[Path]]:
    """
    For a single algo, discover latest timestamp dirs per seed for each cell.
    Returns {cell_name: [run_dir, ...]} where run_dir points at the timestamp folder.
    """
    runs_by_cell: Dict[str, List[Path]] = {}
    for cell_dir in algo_dir.iterdir():
        if not cell_dir.is_dir():
            continue
        cell = cell_dir.name
        if cells and cell not in cells:
            continue
        for seed_dir in cell_dir.iterdir():
            if not seed_dir.is_dir():
                continue
            ts_dir = pick_latest_timestamp_dir(seed_dir)
            if ts_dir is None:
                continue
            runs_by_cell.setdefault(cell, []).append(ts_dir)
    return {k: v for k, v in runs_by_cell.items() if v}


def discover_metrics(run_dirs: List[Path], split: str) -> List[str]:
    names: Set[str] = set()
    for r in run_dirs:
        d = r / split
        if not d.exists():
            continue
        for p in d.iterdir():
            if not (p.is_file() and p.suffix == ".csv"):
                continue
            if try_read_metric_csv(p) is not None:
                names.add(p.stem)
    return sorted(names)


# ---------- Processing ----------


def smooth_series(
    y: np.ndarray, ema: Optional[float], window: Optional[int]
) -> np.ndarray:
    out = y.astype(float).copy()
    if ema is not None:
        a = float(ema)
        if not (0.0 < a < 1.0):
            raise ValueError("--ema must be in (0,1)")
        for i in range(1, len(out)):
            out[i] = a * out[i] + (1 - a) * out[i - 1]
    if window is not None and window > 1:
        k = np.ones(window, dtype=float) / float(window)
        out = np.convolve(out, k, mode="same")
    return out


def interp_to_grid(step: np.ndarray, value: np.ndarray, grid: np.ndarray) -> np.ndarray:
    if len(step) == 0:
        return np.full_like(grid, np.nan, float)
    m = np.isfinite(step) & np.isfinite(value)
    step, value = step[m], value[m]
    if len(step) < 2:
        return np.full_like(grid, np.nan, float)
    i = np.argsort(step)
    step, value = step[i], value[i]
    return np.interp(grid, step, value, left=value[0], right=value[-1])


def z_for_ci(ci: float) -> float:
    table = {50: 0.674, 68: 1.0, 80: 1.282, 90: 1.645, 95: 1.96, 97.5: 2.241, 99: 2.576}
    return table.get(int(ci), 1.96)


def aggregate_runs(
    curves: List[np.ndarray], agg: str, ci: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    X = np.vstack(curves)
    center = np.nanmean(X, axis=0) if agg == "mean" else np.nanmedian(X, axis=0)
    n = np.sum(np.isfinite(X), axis=0).astype(float)
    std = np.nanstd(X, axis=0, ddof=1)
    sem = std / np.sqrt(np.maximum(n, 1.0))
    z = z_for_ci(ci)
    return center, center - z * sem, center + z * sem


# ---------- Plotting ----------


def make_plot(
    title: str,
    grid: np.ndarray,
    curves_by_cell: Dict[str, List[np.ndarray]],
    show_seeds: bool,
    agg: str,
    ci: float,
    xlabel: str,
    ylabel: str,
    outfile: Path,
) -> None:
    plt.figure(figsize=(7.0, 4.2), dpi=200)
    for cell, curves in sorted(curves_by_cell.items()):
        if not curves:
            continue
        if show_seeds and len(curves) > 1:
            for y in curves:
                plt.plot(grid, y, alpha=0.25, linewidth=1.0)
        center, lo, hi = aggregate_runs(curves, agg=agg, ci=ci)
        plt.plot(grid, center, linewidth=2.0, label=cell)
        plt.fill_between(grid, lo, hi, alpha=0.20, linewidth=0)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, linestyle="--", linewidth=0.6, alpha=0.6)
    plt.legend(frameon=False)
    plt.tight_layout()
    outfile.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outfile)
    plt.close()


# ---------- Pipeline ----------


def load_metric_for_run(
    run_dir: Path, split: str, metric: str, ema: Optional[float], window: Optional[int]
) -> Optional[pd.DataFrame]:
    metric_path = run_dir / split / f"{metric}.csv"
    if not metric_path.is_file():
        return None
    df = try_read_metric_csv(metric_path)
    if df is None:
        print(f"[skip] non-plottable: {metric_path}")
        return None
    if ema is not None or (window is not None and window > 1):
        df = df.copy()
        df["value"] = smooth_series(df["value"].to_numpy(), ema=ema, window=window)
    return df


def build_grid(all_steps: List[np.ndarray], bins: int) -> np.ndarray:
    if not all_steps:
        return np.array([])
    maxes = [np.nanmax(s) for s in all_steps if len(s) > 0]
    if not maxes:
        return np.array([])
    xmax = float(np.nanmin(maxes))
    xmin = 0.0
    if xmax <= xmin:
        xmax = float(np.nanmax([np.nanmax(s) for s in all_steps]))
    return np.linspace(xmin, xmax, num=bins)


def main():
    parser = argparse.ArgumentParser(
        description="Batch RL plots: settings × algos × cells × metrics.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--root", type=str, required=True, help="Environment dir (e.g., ./MemoryChain)"
    )
    parser.add_argument(
        "--settings",
        type=str,
        default=None,
        help="Comma list of setting names under --root, or 'all'. If omitted, auto-detect (no settings if algos are directly under --root).",
    )
    parser.add_argument(
        "--algo",
        type=str,
        default=None,
        help="If set, only this algo name under each setting/base",
    )
    parser.add_argument(
        "--cells",
        type=str,
        default=None,
        help="Comma-separated cells to include (e.g., GRUCell,GTrXLCell)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="training",
        help="'training', 'evaluation', 'losses', or 'all'",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        default="episodic_returns",
        help="Comma list or 'auto' to discover all CSVs",
    )
    parser.add_argument("--bins", type=int, default=300)
    parser.add_argument("--ema", type=float, default=None, help="EMA alpha in (0,1)")
    parser.add_argument(
        "--window", type=int, default=None, help="Moving average window (samples)"
    )
    parser.add_argument("--agg", type=str, default="mean", choices=["mean", "median"])
    parser.add_argument("--ci", type=float, default=95.0)
    parser.add_argument("--show-seeds", action="store_true")
    parser.add_argument("--x-millions", dest="x_millions", action="store_true")
    parser.add_argument("--outdir", type=str, default="./plots")
    parser.add_argument(
        "--format", type=str, default="png", choices=["png", "pdf", "svg"]
    )
    parser.add_argument(
        "--out-tree",
        action="store_true",
        help="Nest outputs as OUTDIR[/<setting>]/<algo>/<split>/...",
    )

    args = parser.parse_args()
    root = Path(args.root).expanduser().resolve()
    if not root.exists():
        raise SystemExit(f"Root not found: {root}")

    cell_list = [c.strip() for c in args.cells.split(",")] if args.cells else None
    outdir = Path(args.outdir).expanduser().resolve()
    splits = (
        ["training", "evaluation", "losses"]
        if args.split.lower() == "all"
        else [args.split]
    )

    # ---- Determine settings and algo locations ----
    # If user specified --settings
    if args.settings:
        if args.settings.lower() == "all":
            settings_dirs = sorted([p for p in root.iterdir() if p.is_dir()])
        else:
            wanted = {s.strip() for s in args.settings.split(",")}
            settings_dirs = sorted([root / s for s in wanted if (root / s).is_dir()])
        if not settings_dirs:
            raise SystemExit("No setting directories found under root.")
        layout = [
            (s, discover_algos(s) if args.algo is None else [s / args.algo])
            for s in settings_dirs
        ]
    else:
        # Auto-detect: if algos live directly under root, no settings layer
        algos_at_root = (
            discover_algos(root) if args.algo is None else [root / args.algo]
        )
        algos_at_root = [
            a for a in algos_at_root if a.exists() and looks_like_algo_dir(a)
        ]
        if algos_at_root:
            layout = [(None, algos_at_root)]  # single "no setting" bucket
        else:
            # Treat all immediate subdirs as settings; discover algos within each
            settings_dirs = sorted([p for p in root.iterdir() if p.is_dir()])
            layout = [
                (s, discover_algos(s) if args.algo is None else [s / args.algo])
                for s in settings_dirs
            ]

    # ---- Iterate settings -> algos ----
    for setting_dir, algo_dirs in layout:
        setting_name = None if setting_dir is None else setting_dir.name
        algo_dirs = [a for a in algo_dirs if a.exists() and a.is_dir()]
        if not algo_dirs:
            label = "(root)" if setting_name is None else setting_name
            print(f"[skip] No algo directories in {label}")
            continue

        for algo_dir in algo_dirs:
            if not looks_like_algo_dir(algo_dir):
                print(f"[skip] Not an algo dir: {algo_dir}")
                continue

            runs_by_cell = discover_runs_for_algo(algo_dir, cells=cell_list)
            if not runs_by_cell:
                where = (
                    f"{setting_name}/{algo_dir.name}" if setting_name else algo_dir.name
                )
                print(f"[skip] No runs in {where}")
                continue

            # Metrics to plot
            if args.metrics.lower() == "auto":
                metrics_by_split = {}
                for split in splits:
                    all_run_dirs = [r for rd in runs_by_cell.values() for r in rd]
                    metrics_by_split[split] = discover_metrics(all_run_dirs, split)
                    if not metrics_by_split[split]:
                        where = f"{algo_dir.name}/{split}"
                        if setting_name:
                            where = f"{setting_name}/{where}"
                        print(f"[warn] No metrics discovered in {where}")
            else:
                metric_list = [m.strip() for m in args.metrics.split(",") if m.strip()]
                metrics_by_split = {split: metric_list for split in splits}

            for split in splits:
                metric_list = metrics_by_split.get(split, [])
                if not metric_list:
                    where = f"{algo_dir.name}/{split}"
                    if setting_name:
                        where = f"{setting_name}/{where}"
                    print(f"[skip] {where}: no metrics")
                    continue

                for metric in metric_list:
                    # Load all runs for this metric
                    per_cell_series: Dict[str, List[pd.DataFrame]] = {}
                    for cell, run_dirs in runs_by_cell.items():
                        series = []
                        for r in run_dirs:
                            df = load_metric_for_run(
                                r,
                                split=split,
                                metric=metric,
                                ema=args.ema,
                                window=args.window,
                            )
                            if df is not None and len(df) >= 2:
                                series.append(df)
                        if series:
                            per_cell_series[cell] = series
                    if not per_cell_series:
                        where = f"{algo_dir.name}/{split}/{metric}"
                        if setting_name:
                            where = f"{setting_name}/{where}"
                        print(f"[skip] {where}: no data")
                        continue

                    all_steps = [
                        s["step"].to_numpy()
                        for runs in per_cell_series.values()
                        for s in runs
                    ]
                    grid = build_grid(all_steps, bins=args.bins)
                    if grid.size == 0:
                        where = f"{algo_dir.name}/{split}/{metric}"
                        if setting_name:
                            where = f"{setting_name}/{where}"
                        print(f"[skip] {where}: empty grid")
                        continue

                    curves_by_cell: Dict[str, List[np.ndarray]] = {}
                    for cell, series in per_cell_series.items():
                        curves = []
                        for df in series:
                            y = interp_to_grid(
                                df["step"].to_numpy(float),
                                df["value"].to_numpy(float),
                                grid,
                            )
                            curves.append(y)
                        curves_by_cell[cell] = curves

                    if args.x_millions:
                        x = grid / 1e6
                        xlabel = "Environment steps (millions)"
                    else:
                        x = grid
                        xlabel = "Environment steps"
                    ylabel = metric.replace("_", " ").capitalize()

                    # Titles / filenames include setting if present
                    if setting_name:
                        title = f"{root.name}/{setting_name} · {algo_dir.name} · {split} · {metric}"
                        fname = f"{root.name}_{setting_name}_{algo_dir.name}_{split}_{metric}.{args.format}"
                    else:
                        title = f"{root.name} · {algo_dir.name} · {split} · {metric}"
                        fname = f"{root.name}_{algo_dir.name}_{split}_{metric}.{args.format}"

                    # Output path
                    if args.out_tree:
                        if setting_name:
                            outpath = (
                                outdir / setting_name / algo_dir.name / split / fname
                            )
                        else:
                            outpath = outdir / algo_dir.name / split / fname
                    else:
                        outpath = outdir / fname

                    make_plot(
                        title=title,
                        grid=x,
                        curves_by_cell=curves_by_cell,
                        show_seeds=args.show_seeds,
                        agg=args.agg,
                        ci=args.ci,
                        xlabel=xlabel,
                        ylabel=ylabel,
                        outfile=outpath,
                    )
                    where = f"{algo_dir.name}/{split}/{metric}"
                    if setting_name:
                        where = f"{setting_name}/{where}"
                    print(f"[ok] {where} → {outpath}")


if __name__ == "__main__":
    main()
