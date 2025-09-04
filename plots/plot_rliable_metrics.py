"""
Plot RL metrics using rliable with robust aggregation across seeds and nice visuals.

Usage (example):
  pip install -U rliable matplotlib pandas numpy seaborn
  python plot_with_rliable.py \
    --root /path/to/Craftax-Symbolic-v1/rppo \
    --out ./plots \
    --reps 20000

What you get (saved in --out):
  - Raw per-seed curves for each metric (value vs. steps).
  - IQM sample-efficiency curves with stratified bootstrap 95% CIs (via rliable).
  - Performance profiles at the final step for return-like metrics.
  - Aggregate bar charts (Median, IQM, Mean, Optimality Gap) at the final step.
  - (Optional) Probability-of-improvement matrix at the final step.
"""
import argparse
import json
import math
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path

# rliable imports
from rliable import library as rly
from rliable import metrics
from rliable import plot_utils

sns.set_context("talk")

# -----------------------------
# Data model & discovery
# -----------------------------

@dataclass
class Series:
    steps: np.ndarray  # shape (T,)
    values: np.ndarray  # shape (T,)

@dataclass
class RunMetric:
    algo: str
    seed: str
    kind: str  # e.g., "training", "evaluation", "losses"
    name: str  # e.g., "episodic_returns", "actor_loss"
    series: Series

MetricDict = Dict[str, Dict[str, Dict[str, Dict[str, Series]]]]
# metric_dict[algo][seed][kind][name] = Series

CSV_ALIASES = {
    "episodic_returns.csv": "episodic_returns",
    "episodic_lengths.csv": "episodic_lengths",
    "discount.csv": "discount",
    "actor_loss.csv": "actor_loss",
    "critic_loss.csv": "critic_loss",
    "approx_kl.csv": "approx_kl",
    "clipfrac.csv": "clipfrac",
    "entropy.csv": "entropy",
}

RETURNS_LIKE = {"episodic_returns"}  # "higher is better"
LOWER_BETTER = {"episodic_lengths", "actor_loss", "critic_loss", "approx_kl", "clipfrac"}  # invert for rliable
NEUTRAL = {"entropy", "discount"}  # plotted raw + IQM curves (no perf profile)

def is_numeric(x: str) -> bool:
    try:
        float(x)
        return True
    except Exception:
        return False

def read_csv_as_series(fp: Path) -> Series:
    """
    Robustly read a two-column CSV (steps, value) whether it has a header or not.
    """
    try:
        df = pd.read_csv(fp, header=None)
        if df.shape[1] < 2:
            # Try again assuming comma separator with potential header
            df = pd.read_csv(fp)
        # Heuristic: identify the step/value columns
        # If first row looks like header strings, try names=['step','value'].
        cols = list(df.columns)
        if len(cols) >= 2 and (is_numeric(str(cols[0])) or is_numeric(str(cols[1]))):
            # Headers were actually data; reload with no header
            df = pd.read_csv(fp, header=None, names=["step", "value"])
        else:
            # Use first two columns and rename
            df = df.iloc[:, :2]
            df.columns = ["step", "value"]
    except Exception:
        # Final fallback
        df = pd.read_csv(fp, header=None, names=["step", "value"])
    # Drop NA, sort by step, keep unique steps (last occurrence wins)
    df = df.dropna().copy()
    df = df.sort_values("step")
    df = df.drop_duplicates(subset=["step"], keep="last")
    steps = df["step"].to_numpy(dtype=float)
    vals = df["value"].to_numpy(dtype=float)
    return Series(steps=steps, values=vals)

def discover(root: Path) -> MetricDict:
    """
    Walk the given root (e.g., Craftax-Symbolic-v1/rppo) and collect all metrics.
    Expects subdirs: <algo>/<seed>/<timestamp>/{training,evaluation,losses}/*.csv
    """
    metric_dict: MetricDict = {}
    for algo_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
        algo = algo_dir.name
        metric_dict.setdefault(algo, {})
        for seed_dir in sorted([p for p in algo_dir.iterdir() if p.is_dir()]):
            seed = seed_dir.name
            # pick latest timestamp dir if multiple
            run_dirs = sorted([p for p in seed_dir.iterdir() if p.is_dir()])
            if not run_dirs:
                continue
            run_dir = run_dirs[-1]
            for kind in ("training", "evaluation", "losses"):
                sub = run_dir / kind
                if not sub.exists():
                    continue
                for csv_fp in sorted(sub.glob("*.csv")):
                    name = CSV_ALIASES.get(csv_fp.name, csv_fp.stem)
                    series = read_csv_as_series(csv_fp)
                    metric_dict[algo].setdefault(seed, {}).setdefault(kind, {})[name] = series
    return metric_dict

# -----------------------------
# Utilities
# -----------------------------

def make_grid(series_list: List[Series], max_points: int = 400) -> np.ndarray:
    """
    Build a shared step grid across multiple Series.
    Union of steps, possibly downsampled to at most max_points points.
    """
    if not series_list:
        return np.array([])
    all_steps = np.unique(np.concatenate([s.steps for s in series_list]))
    if all_steps.size > max_points:
        # Evenly spaced downsampling
        idx = np.linspace(0, all_steps.size - 1, max_points).round().astype(int)
        all_steps = all_steps[idx]
    return all_steps

def interp_to_grid(series: Series, grid: np.ndarray) -> np.ndarray:
    """
    Interpolate series values onto the grid. Extrapolation uses nearest edge value.
    """
    if series.steps.size == 0:
        return np.full_like(grid, np.nan, dtype=float)
    # Clip grid to bounds to avoid NaNs, then fill edges with nearest
    y = np.interp(np.clip(grid, series.steps[0], series.steps[-1]), series.steps, series.values)
    # For left tail: constant at first value; right tail: constant at last value
    y[grid < series.steps[0]] = series.values[0]
    y[grid > series.steps[-1]] = series.values[-1]
    return y

def build_scores_3d(metric_dict: MetricDict, kind: str, name: str, max_points: int = 400
                   ) -> Tuple[np.ndarray, Dict[str, np.ndarray], np.ndarray]:
    """
    Build a grid and per-algo 3D scores array compatible with rliable's sample-efficiency plotting.

    Returns:
      grid: shape (T,) step values.
      per_algo: dict algo -> scores (num_seeds x num_tasks x T), here num_tasks=1.
      valid_mask: for each algo, we still interpolate; mask not needed with our interpolation, but
                  returned for potential future use.
    """
    per_algo = {}
    # Gather all series to make a global grid for this metric
    all_series: List[Series] = []
    for algo, seeds in metric_dict.items():
        for seed, groups in seeds.items():
            if kind in groups and name in groups[kind]:
                all_series.append(groups[kind][name])
    grid = make_grid(all_series, max_points=max_points)
    if grid.size == 0:
        return grid, per_algo, np.array([])
    # Interpolate per algo/seed
    for algo, seeds in metric_dict.items():
        per_seed: List[np.ndarray] = []
        for seed, groups in seeds.items():
            if kind in groups and name in groups[kind]:
                y = interp_to_grid(groups[kind][name], grid)  # (T,)
                per_seed.append(y[None, None, :])  # (1, 1, T)
        if per_seed:
            arr = np.concatenate(per_seed, axis=0)  # (num_seeds, 1, T)
            per_algo[algo] = arr
    return grid, per_algo, np.array([])

# -----------------------------
# rliable aggregators (nan-safe)
# -----------------------------

def _nan_safe(fn):
    def wrapper(scores_2d: np.ndarray) -> np.ndarray:
        # scores_2d expected shape: (num_runs x num_tasks)
        # Remove rows with all NaNs
        if scores_2d.ndim != 2:
            raise ValueError("Expected 2D array for aggregator.")
        mask_rows = ~np.all(~np.isfinite(scores_2d), axis=1)
        scores = scores_2d[mask_rows]
        if scores.size == 0:
            return np.array([np.nan])
        return fn(scores)
    return wrapper

aggregate_median = _nan_safe(lambda x: np.array([metrics.aggregate_median(x)]))
aggregate_iqm    = _nan_safe(lambda x: np.array([metrics.aggregate_iqm(x)]))
aggregate_mean   = _nan_safe(lambda x: np.array([metrics.aggregate_mean(x)]))
aggregate_optgap = _nan_safe(lambda x: np.array([metrics.aggregate_optimality_gap(x)]))

# -----------------------------
# Plotting
# -----------------------------

def ensure_outdir(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)

def savefig(fig, out: Path, name: str, dpi: int = 140):
    fig.tight_layout()
    fp = out / f"{name}.png"
    fig.savefig(fp, dpi=dpi, bbox_inches="tight")
    print(f"Saved: {fp}")

def plot_raw_curves(metric_dict: MetricDict, kind: str, name: str, out: Path, dpi: int = 140):
    fig, ax = plt.subplots(figsize=(9, 6))
    for algo, seeds in metric_dict.items():
        for seed, groups in seeds.items():
            if kind in groups and name in groups[kind]:
                s = groups[kind][name]
                ax.plot(s.steps, s.values, alpha=0.6, label=f"{algo} (seed {seed})")
    ax.set_title(f"{kind}/{name}: per-seed curves")
    ax.set_xlabel("Steps")
    ax.set_ylabel("Value")
    # Avoid duplicate labels
    handles, labels = ax.get_legend_handles_labels()
    if len(labels) > 20:  # don't spam
        ax.legend().remove()
    else:
        ax.legend(ncol=2, fontsize="small")
    savefig(fig, out, f"{kind}_{name}__raw", dpi=dpi)
    plt.close(fig)

def plot_iqm_curve(metric_dict: MetricDict, kind: str, name: str, out: Path, reps: int, dpi: int = 140):
    grid, per_algo, _ = build_scores_3d(metric_dict, kind, name)
    if grid.size == 0 or not per_algo:
        return
    algorithms = list(per_algo.keys())
    # Build dict as expected by rliable.get_interval_estimates: algo -> (runs x tasks x T)
    scores_dict = {algo: per_algo[algo] for algo in algorithms}
    # Aggregator across steps (returns vector of length T)
    def iqm_over_steps(scores_3d: np.ndarray) -> np.ndarray:
        # scores_3d: (num_runs, num_tasks=1, T)
        # For each t, take slice (num_runs x num_tasks) and compute IQM
        T = scores_3d.shape[-1]
        vals = []
        for t in range(T):
            vals.append(metrics.aggregate_iqm(scores_3d[..., t]))
        return np.array(vals)

    
    iqm_scores, iqm_cis = rly.get_interval_estimates(scores_dict, iqm_over_steps, reps=reps)
    ret = plot_utils.plot_sample_efficiency_curve(
        grid, iqm_scores, iqm_cis, algorithms=algorithms,
        xlabel="Steps", ylabel="IQM (higher is better)")
    # Handle various return types (fig or (fig, ax) or None)
    fig = None
    if hasattr(ret, "savefig"):
        fig = ret
    elif isinstance(ret, tuple) and ret and hasattr(ret[0], "savefig"):
        fig = ret[0]
    else:
        import matplotlib.pyplot as _plt
        fig = _plt.gcf()
    savefig(fig, out, f"{kind}_{name}__iqm_curve", dpi=dpi)
    _plt.close(fig)


def plot_aggregate_bars_at_final(metric_dict: MetricDict, kind: str, name: str, out: Path, reps: int, dpi: int = 140, invert: bool = False):
    # Collect final-step values per algo across seeds -> (runs x tasks=1)
    algo_to_scores = {}
    for algo, seeds in metric_dict.items():
        vals = []
        for seed, groups in seeds.items():
            if kind in groups and name in groups[kind]:
                s = groups[kind][name]
                if s.values.size == 0:
                    continue
                v = s.values[-1]  # last step value
                vals.append([(-v if invert else v)])
        if vals:
            algo_to_scores[algo] = np.array(vals)  # shape (num_runs, 1)
    if not algo_to_scores:
        return
    aggregate_func = lambda x: np.array([
        metrics.aggregate_median(x),
        metrics.aggregate_iqm(x),
        metrics.aggregate_mean(x),
        metrics.aggregate_optimality_gap(x),
    ])
    agg_scores, agg_cis = rly.get_interval_estimates(algo_to_scores, aggregate_func, reps=reps)
    fig, axes = plot_utils.plot_interval_estimates(
        agg_scores, agg_cis,
        algorithms=list(algo_to_scores.keys()),
        metric_names=["Median", "IQM", "Mean", "Optimality Gap"],
        xlabel=("Inverted value" if invert else "Value"))
    savefig(fig, out, f"{kind}_{name}__aggregate_bars_final", dpi=dpi)
    plt.close(fig)

def plot_performance_profile_at_final(metric_dict: MetricDict, kind: str, name: str, out: Path, reps: int, dpi: int = 140, invert: bool = False):
    # Prepare dict algo -> (runs x tasks=1) of final-step values
    algo_to_scores = {}
    for algo, seeds in metric_dict.items():
        vals = []
        for seed, groups in seeds.items():
            if kind in groups and name in groups[kind]:
                s = groups[kind][name]
                if s.values.size == 0:
                    continue
                v = s.values[-1]
                vals.append([(-v if invert else v)])
        if vals:
            algo_to_scores[algo] = np.array(vals)  # (num_runs, 1)
    if not algo_to_scores:
        return
    # Choose thresholds from pooled scores
    pooled = np.concatenate(list(algo_to_scores.values()), axis=0).flatten()
    lo, hi = np.nanmin(pooled), np.nanmax(pooled)
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        return
    thresholds = np.linspace(lo, hi, 101)
    prof, prof_cis = rly.create_performance_profile(algo_to_scores, thresholds)
    fig, ax = plt.subplots(ncols=1, figsize=(8, 6))
    plot_utils.plot_performance_profiles(
        prof, thresholds,
        performance_profile_cis=prof_cis,
        xlabel=("Inverted value (higher is better)" if invert else "Value"),
        ax=ax)
    ax.set_title(f"Performance profile at final step: {kind}/{name}")
    savefig(fig, out, f"{kind}_{name}__performance_profile_final", dpi=dpi)
    plt.close(fig)

def plot_prob_improvement_at_final(metric_dict: MetricDict, kind: str, name: str, out: Path, reps: int, dpi: int = 140, invert: bool = False):
    # Build pair dict for rliable: {"A vs B": (A_scores, B_scores)}
    finals = {}
    for algo, seeds in metric_dict.items():
        vals = []
        for seed, groups in seeds.items():
            if kind in groups and name in groups[kind]:
                s = groups[kind][name]
                if s.values.size == 0:
                    continue
                v = s.values[-1]
                vals.append(-v if invert else v)
        if vals:
            finals[algo] = np.array(vals, dtype=float)[:, None]  # (runs, 1)
    if len(finals) < 2:
        return
    pairs = {}
    algos = list(finals.keys())
    for i in range(len(algos)):
        for j in range(i + 1, len(algos)):
            a, b = algos[i], algos[j]
            pairs[f"{a} vs {b}"] = (finals[a], finals[b])
    avg_probs, avg_prob_cis = rly.get_interval_estimates(pairs, metrics.probability_of_improvement, reps=reps)
    fig = plot_utils.plot_probability_of_improvement(avg_probs, avg_prob_cis)
    savefig(fig, out, f"{kind}_{name}__prob_improvement_final", dpi=dpi)
    plt.close(fig)

# -----------------------------
# Driver
# -----------------------------

def main():
    parser = argparse.ArgumentParser(description="Plot metrics with rliable.")
    parser.add_argument("--root", type=Path, required=True,
                        help="Path to .../Craftax-Symbolic-v1/rppo")
    parser.add_argument("--out", type=Path, default=Path("./plots"), help="Output dir")
    parser.add_argument("--reps", type=int, default=20000, help="Bootstrap repetitions")
    parser.add_argument("--dpi", type=int, default=140, help="Figure DPI")
    parser.add_argument("--max_points", type=int, default=400, help="Max x-points per curve")
    parser.add_argument("--skip_prob_improvement", action="store_true",
                        help="Skip probability-of-improvement plots")
    args = parser.parse_args()

    ensure_outdir(args.out)
    metric_dict = discover(args.root)

    # Enumerate known metrics
    entries: List[Tuple[str, str]] = []  # (kind, name)
    for algo, seeds in metric_dict.items():
        for seed, groups in seeds.items():
            for kind, m in groups.items():
                for name in m.keys():
                    entries.append((kind, name))
    entries = sorted(set(entries))

    # Plot everything
    for kind, name in entries:
        print(f"Processing {kind}/{name}")

        # 1) Raw per-seed curves (always)
        plot_raw_curves(metric_dict, kind, name, args.out, dpi=args.dpi)

        # 2) IQM sample-efficiency curve (always)
        plot_iqm_curve(metric_dict, kind, name, args.out, reps=args.reps, dpi=args.dpi)

        # 3) Aggregate bars + Performance profile + Probability of improvement at final step
        invert = name in LOWER_BETTER
        if name in RETURNS_LIKE or name in LOWER_BETTER or name in NEUTRAL:
            plot_aggregate_bars_at_final(metric_dict, kind, name, args.out, reps=args.reps, dpi=args.dpi, invert=invert)
        if name in RETURNS_LIKE or name in LOWER_BETTER:
            plot_performance_profile_at_final(metric_dict, kind, name, args.out, reps=args.reps, dpi=args.dpi, invert=invert)
            if not args.skip_prob_improvement:
                plot_prob_improvement_at_final(metric_dict, kind, name, args.out, reps=args.reps, dpi=args.dpi, invert=invert)

if __name__ == "__main__":
    main()

