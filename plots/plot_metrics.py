"""
crawl logs/<env>/<algorithm>/<cell>/<seed>/<timestamp>/
and generate professional reports per metric using rliable.

Outputs (per category & metric):
  - report.pdf: multi-page:
      1) Aggregate interval estimates (Median, IQM, Mean, Optimality Gap)
      2) Performance profiles (+ CIs)
      3) Sample-efficiency curves (IQM over steps with CIs)
      4) Probability of improvement heatmap (pairwise algorithms)
  - aggregate_summary.csv: point estimates + CIs per algorithm
  - prob_improvement.csv: avg probability of improvement for all pairs
  - steps_iqm.csv: IQM over steps (+ CIs) per algorithm

Assumptions:
  - CSVs contain a 'step' column and ≥1 numeric metric columns.
  - We treat 'environment' as the "task" dimension for rliable.
  - We aggregate across seeds (runs) and timestamps for each (env, alg, cell).
  - Subfolders under evaluation/training/losses are searched recursively.

Example:
  python plot_metrics.py --root ./logs --out ./plots --reps 50000
"""

import argparse
import itertools
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm

# rliable
from rliable import library as rly
from rliable import metrics
from rliable import plot_utils

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_reports.py — crawl logs/<env>/<algorithm>/<cell>/<seed>/<timestamp>/
and generate professional reports per metric using rliable.

Outputs (per category & metric):
  - report.pdf: multi-page:
      1) Aggregate interval estimates (Median, IQM, Mean, Optimality Gap)
      2) Performance profiles (+ CIs)
      3) Sample-efficiency curves (IQM over steps with CIs)
      4) Probability of improvement heatmap (pairwise algorithms)
  - aggregate_summary.csv: point estimates + CIs per algorithm
  - prob_improvement.csv: avg probability of improvement for all pairs
  - steps_iqm.csv: IQM over steps (+ CIs) per algorithm

Assumptions:
  - CSVs contain a 'step' column and ≥1 numeric metric columns.
  - We treat 'environment' as the "task" dimension for rliable.
  - We aggregate across seeds (runs) and timestamps for each (env, alg, cell).
  - Subfolders under evaluation/training/losses are searched recursively.

Example:
  python make_reports.py --root ./logs --out ./reports --reps 50000
"""

import argparse
import itertools
import math
import os
from pathlib import Path
from typing import Dict, List, Tuple, DefaultDict
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm

# rliable
from rliable import library as rly
from rliable import metrics
from rliable import plot_utils

# -------------------------- CLI ---------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Generate rliable reports from logs.")
    p.add_argument("--root", type=Path, required=True,
                   help="Root folder: logs/<env>/<algorithm>/<cell>/<seed>/<timestamp>/")
    p.add_argument("--out", type=Path, default=Path("./reports"),
                   help="Output directory for reports.")
    p.add_argument("--reps", type=int, default=50000,
                   help="Bootstrap repetitions for CIs (stratified).")
    p.add_argument("--ci", type=float, default=0.95, help="Confidence level.")
    p.add_argument("--min_runs", type=int, default=1,
                   help="Require at least this many runs (seeds) per (env,alg) to include.")
    p.add_argument("--max_steps", type=int, default=200,
                   help="Max standardized steps for sample-efficiency curves.")
    p.add_argument("--dpi", type=int, default=180, help="Figure DPI.")
    p.add_argument("--style", type=str, default="whitegrid", help="Seaborn style.")
    p.add_argument("--format", nargs="+", default=["pdf"], choices=["pdf", "png", "svg"])
    return p.parse_args()

# -------------------------- Files & Loading ---------------------------------

CATEGORY_NAMES = ("evaluation", "training", "losses")

def _ci_rows_first(ci: np.ndarray) -> np.ndarray:
    """Return CI as shape (N, 2): [low, high] per row."""
    ci = np.asarray(ci)
    if ci.ndim != 2:
        return ci
    if ci.shape[1] == 2:   # already (N,2)
        return ci
    if ci.shape[0] == 2:   # currently (2,N)
        return ci.T
    return ci


def find_csvs(root: Path) -> List[Tuple[str, str, str, str, str, str, Path]]:
    """
    Return list of (env, alg, cell, seed, ts, category, path) for all CSVs found
    under .../<env>/<algorithm>/<cell>/<seed>/<timestamp>/<category>/.../*.csv
    """
    results = []
    if not root.exists():
        return results
    # Expect depth shape; walk environments first
    for env_dir in root.iterdir():
        if not env_dir.is_dir():
            continue
        env = env_dir.name
        for alg_dir in env_dir.iterdir():
            if not alg_dir.is_dir():
                continue
            alg = alg_dir.name
            for cell_dir in alg_dir.iterdir():
                if not cell_dir.is_dir():
                    continue
                cell = cell_dir.name
                for seed_dir in cell_dir.iterdir():
                    if not seed_dir.is_dir():
                        continue
                    seed = seed_dir.name
                    for ts_dir in seed_dir.iterdir():
                        if not ts_dir.is_dir():
                            continue
                        ts = ts_dir.name
                        for category in CATEGORY_NAMES:
                            cat_dir = ts_dir / category
                            if not cat_dir.exists():
                                continue
                            # recursive glob for CSVs
                            for path in cat_dir.rglob("*.csv"):
                                results.append((env, alg, cell, seed, ts, category, path))
    return results

# --- add near imports ---
import unicodedata, re

def safe_metric_name(s: str) -> str:
    # slug for filesystem safety
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    s = s.strip().replace(" ", "_")
    s = re.sub(r"[^\w\-.]+", "_", s)       # keep [A-Za-z0-9_-.]
    s = re.sub(r"_{2,}", "_", s).strip("_")
    return s or "metric"

def _looks_like_header(cols) -> bool:
    for c in cols:
        t = str(c).strip().lower()
        if t == "step":
            return True
        # treat anything with letters/underscores as a header token
        if re.search(r"[a-z_]", t):
            return True
        # if purely numeric, keep checking others
        try:
            float(t)
        except ValueError:
            return True
    return False

def read_metrics_from_csv(path: Path) -> pd.DataFrame:
    """
    Robust CSV reader:
      - Detects missing header; if absent, names are: step, <filename>[,_1,_2...]
      - Coerces to numeric; drops rows missing step; sorts by step.
      - Drops rows where all metric columns are NaN (e.g., dangling partial line).
    """
    base = path.stem  # e.g., episodic_returns.csv -> episodic_returns
    with open(path, "r", newline="") as f:
        first = f.readline().strip()
    header_present = _looks_like_header(first.split(","))

    if header_present:
        df = pd.read_csv(path)
        # common junk index column
        df = df.loc[:, ~df.columns.str.match(r"^Unnamed:")]
        if "step" not in df.columns:
            # If someone wrote header but forgot 'step', assume first col is step
            df = df.rename(columns={df.columns[0]: "step"})
    else:
        df = pd.read_csv(path, header=None)
        if df.shape[1] == 1:
            # one-column file -> implicit step
            df.columns = [base]
            df.insert(0, "step", np.arange(len(df), dtype=np.int64))
        else:
            # multi-col file: first column is steps, rest are metrics
            names = ["step"]
            if df.shape[1] == 2:
                names.append(base)
            else:
                names += [f"{base}_{i}" for i in range(1, df.shape[1])]
            df.columns = names

    # numeric coercion
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # clean rows
    metric_cols = [c for c in df.columns if c != "step"]
    df = df.dropna(subset=["step"])
    # drop rows where all metrics are NaN (e.g., trailing partial lines)
    if metric_cols:
        df = df.dropna(subset=metric_cols, how="all")

    # finalize
    df = df.sort_values("step").reset_index(drop=True)
    return df

# -------------------------- Data Structuring --------------------------------

class RunSeries:
    """Holds a single run's time series for one metric."""
    def __init__(self, steps: np.ndarray, values: np.ndarray):
        self.steps = steps.astype(np.float64)
        self.values = values.astype(np.float64)

def collect_data(root: Path):
    """
    Returns nested dict:
      data[category][metric][algorithm_label]['by_env'][env] -> list[RunSeries]
    where algorithm_label = f"{algorithm}/{cell}" (cell folded into label).
    """
    files = find_csvs(root)
    data = {cat: defaultdict(lambda: defaultdict(lambda: {"by_env": defaultdict(list)}))
            for cat in CATEGORY_NAMES}

    for env, alg, cell, seed, ts, category, path in tqdm(files, desc="Scanning CSVs"):
        df = read_metrics_from_csv(path)
        metric_cols = [c for c in df.columns if c != "step"]
        if not metric_cols:
            continue
        alg_label = f"{alg}/{cell}"
        steps = df["step"].values
        for m in metric_cols:
            series = RunSeries(steps=steps, values=df[m].values)
            data[category][m][alg_label]["by_env"][env].append(series)
    return data

# -------------------------- Helpers: interpolation & grids -------------------

def standardize_step_grid(run_lists_per_env: Dict[str, List[RunSeries]], max_steps=200):
    """
    Build a common step grid across environments and interpolate each run onto it.
    Returns:
      grid: np.ndarray [S]
      runs_per_env: dict env -> list[np.ndarray of shape [S]]
    """
    # 1) Find global min/max step
    mins, maxs = [], []
    for env, runs in run_lists_per_env.items():
        for r in runs:
            if len(r.steps):
                mins.append(r.steps.min())
                maxs.append(r.steps.max())
    if not mins:
        return None, {}
    lo, hi = min(mins), max(maxs)
    if hi <= lo:
        hi = lo + 1.0
    # 2) Uniform grid of up to max_steps
    S = min(max_steps, int(hi - lo + 1)) if (hi - lo + 1) >= 2 else max_steps
    grid = np.linspace(lo, hi, num=S)
    # 3) Interpolate each run
    runs_per_env_interp = {}
    for env, runs in run_lists_per_env.items():
        interp_arrays = []
        for r in runs:
            # guard against duplicates in steps
            steps_unique, idx = np.unique(r.steps, return_index=True)
            vals_unique = r.values[idx]
            y = np.interp(grid, steps_unique, vals_unique)
            interp_arrays.append(y)
        runs_per_env_interp[env] = interp_arrays
    return grid, runs_per_env_interp

def final_values_matrix(run_lists_per_env: Dict[str, List[RunSeries]]) -> Tuple[np.ndarray, List[str]]:
    """
    Build a matrix of shape (num_runs, num_envs) by taking the final value per run.
    Different envs can have different #runs; we pad with NaN and then drop rows that
    are all-NaN. rliable metrics are computed per algorithm independently; we’ll
    nanmask where needed.
    """
    envs = sorted(run_lists_per_env.keys())
    columns = envs
    # Collect per env list of finals
    finals_per_env = []
    max_runs = 0
    for env in envs:
        finals = []
        for r in run_lists_per_env[env]:
            if len(r.values):
                finals.append(r.values[-1])
        finals_per_env.append(np.array(finals, dtype=np.float64))
        max_runs = max(max_runs, len(finals))
    if max_runs == 0:
        return np.zeros((0, len(envs))), columns
    # Pad to rectangular
    mats = []
    for finals in finals_per_env:
        if len(finals) < max_runs:
            pad = np.full((max_runs - len(finals),), np.nan)
            finals = np.concatenate([finals, pad], axis=0)
        mats.append(finals.reshape(-1, 1))
    mat = np.concatenate(mats, axis=1)  # [max_runs, num_envs]
    # Drop all-NaN rows
    mask = ~np.all(np.isnan(mat), axis=1)
    return mat[mask], columns

# -------------------------- rliable Wrappers --------------------------------

def compute_aggregate_estimates(scores_dict: Dict[str, np.ndarray], reps: int):
    """
    scores_dict: alg -> (runs x tasks) array with NaNs allowed
    Returns aggregate_scores, aggregate_cis, metric_names
    """
    def aggregate_func(x):
        # x is (runs x tasks). rliable metrics expect finite numbers; drop NaNs.
        # We’ll mask per-call to avoid bias.
        x = np.array(x, dtype=np.float64)
        if np.isnan(x).any():
            # Replace with masked version per task
            cols = []
            for j in range(x.shape[1]):
                col = x[:, j]
                col = col[~np.isnan(col)]
                if len(col) == 0:
                    col = np.array([np.nan])
                cols.append(col)
            # ragged -> pad to max
            m = max(len(c) for c in cols)
            X = np.full((m, len(cols)), np.nan)
            for j, c in enumerate(cols):
                X[:len(c), j] = c
            x = X

        return np.array([
            metrics.aggregate_median(x),
            metrics.aggregate_iqm(x),
            metrics.aggregate_mean(x),
            metrics.aggregate_optimality_gap(x)
        ])
    aggregate_scores, aggregate_cis = rly.get_interval_estimates(
        scores_dict, aggregate_func, reps=reps)
    metric_names = ['Median', 'IQM', 'Mean', 'Optimality Gap']
    return aggregate_scores, aggregate_cis, metric_names

def compute_performance_profiles(scores_dict: Dict[str, np.ndarray], num_thresh=81):
    # Build thresholds across all algorithms/tasks
    all_vals = np.concatenate(
        [v.flatten()[~np.isnan(v.flatten())] for v in scores_dict.values() if v.size], axis=0
    ) if scores_dict else np.array([])
    if all_vals.size == 0:
        return None, None, None
    lo, hi = np.nanmin(all_vals), np.nanpercentile(all_vals, 99.0)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        hi = lo + 1.0
    thresholds = np.linspace(lo, hi, num_thresh)
    perf_prof, perf_prof_cis = rly.create_performance_profile(scores_dict, thresholds)
    return perf_prof, perf_prof_cis, thresholds

def build_sample_efficiency_inputs(interp_runs_dict: Dict[str, List[np.ndarray]]) -> Tuple[np.ndarray, List[str]]:
    """
    Convert env -> [runs arrays [S]] into array (num_runs x num_envs x S) by
    padding per-env run counts to max and dropping all-NaN runs.
    """
    envs = sorted(interp_runs_dict.keys())
    if not envs:
        return np.zeros((0, 0, 0)), envs
    max_runs = max(len(v) for v in interp_runs_dict.values())
    S = len(next(iter(interp_runs_dict.values()))[0]) if max_runs and interp_runs_dict[envs[0]] else 0
    if max_runs == 0 or S == 0:
        return np.zeros((0, 0, 0)), envs

    # Build [max_runs, num_envs, S] with NaNs, then mask rows that are all-NaN
    arr = np.full((max_runs, len(envs), S), np.nan, dtype=np.float64)
    for j, env in enumerate(envs):
        runs = interp_runs_dict[env]
        for i, y in enumerate(runs):
            arr[i, j, :] = y
    # Drop runs that are NaN across all envs
    keep = ~np.all(np.isnan(arr), axis=(1, 2))
    return arr[keep], envs

def compute_iqm_over_steps(arr_runs_env_steps: np.ndarray, reps: int):
    """
    arr_runs_env_steps: shape (R, E, S).
    Returns iqm_scores [S, A] via dict-of-alg later; and cis [S, 2, A] via rliable.
    We compute IQM across runs x envs for each step.
    """
    if arr_runs_env_steps.size == 0:
        return None, None
    S = arr_runs_env_steps.shape[-1]
    def iqm_func(x):
        # x: (R, E, S)
        vals = []
        for s in range(S):
            xs = x[..., s]
            # flatten runs×envs, drop NaN
            flat = xs.reshape(-1)
            flat = flat[~np.isnan(flat)]
            if flat.size == 0:
                vals.append(np.nan)
            else:
                # rliable expects (runs x tasks); hack: treat as (n,1)
                vals.append(metrics.aggregate_iqm(flat.reshape(-1, 1)))
        return np.array(vals)
    iqm_scores, iqm_cis = rly.get_interval_estimates(
        {"alg": arr_runs_env_steps}, iqm_func, reps=reps)
    # Unwrap single-key dict
    return iqm_scores["alg"], iqm_cis["alg"]

def prob_improvement_pairs(scores_dict: Dict[str, np.ndarray], reps: int) -> Tuple[pd.DataFrame, Dict[Tuple[str,str], Tuple[np.ndarray, np.ndarray]]]:
    """
    Compute probability of improvement for all ordered pairs (A,B).
    Returns:
      table: DataFrame (algorithms x algorithms)
      cis_dict: (A,B) -> (estimate, ci)
    """
    algs = sorted(scores_dict.keys())
    pair_dict = {}
    for a, b in itertools.permutations(algs, 2):
        Xa, Xb = scores_dict[a], scores_dict[b]
        # Align to common number of runs per task by truncation
        if Xa.size == 0 or Xb.size == 0:
            continue
        runs_a, tasks_a = Xa.shape
        runs_b, tasks_b = Xb.shape
        tasks = min(tasks_a, tasks_b)
        if tasks == 0:
            continue
        R = min(runs_a, runs_b)
        Xa_ = Xa[:R, :tasks]
        Xb_ = Xb[:R, :tasks]
        pair_dict[(a, b)] = (Xa_, Xb_)

    if not pair_dict:
        return pd.DataFrame(index=algs, columns=algs, dtype=float), {}

    # rliable expects dict: name -> (Xa, Xb)
    in_dict = {f"{a}__vs__{b}": (Xa, Xb) for (a, b), (Xa, Xb) in pair_dict.items()}
    avg_probs, avg_prob_cis = rly.get_interval_estimates(in_dict, metrics.probability_of_improvement, reps=reps)

    # Build square table
    table = pd.DataFrame(index=algs, columns=algs, dtype=float)
    cis_out = {}
    for key, val in avg_probs.items():
        a, b = key.split("__vs__")
        table.loc[a, b] = float(val)
        cis_out[(a, b)] = (avg_probs[key], avg_prob_cis[key])
    return table, cis_out

# -------------------------- Plotting ----------------------------------------

def set_style(style="whitegrid"):
    sns.set_style(style)
    plt.rcParams.update({
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.constrained_layout.use": True,  # use constrained layout, not autolayout
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        # Embed TrueType → better PDF text and symbols (τ)
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

def page_title(fig, title: str, subtitle: str = ""):
    fig.suptitle(title, fontsize=14)

def plot_interval_estimates_page(aggregate_scores, aggregate_cis, metric_names, algorithms, xlabel, pdf, title):
    fig, axes = plot_utils.plot_interval_estimates(
        aggregate_scores, aggregate_cis,
        metric_names=metric_names, algorithms=algorithms, xlabel=xlabel)
    fig.set_size_inches(8.8, 5.6)
    page_title(fig, title)
    pdf.savefig(fig)
    plt.close(fig)

def plot_performance_profiles_page(perf_prof, thresholds, perf_prof_cis, algorithms, xlabel, pdf, title):
    fig, ax = plt.subplots(figsize=(9.6, 5.8), constrained_layout=True)
    plot_utils.plot_performance_profiles(
        perf_prof, thresholds, performance_profile_cis=perf_prof_cis, xlabel=xlabel, ax=ax)
    ax.legend(title="Algorithm", loc="upper left", bbox_to_anchor=(1.02, 1.0))
    page_title(fig, title)
    pdf.savefig(fig)
    plt.close(fig)


from matplotlib.ticker import ScalarFormatter

def plot_sample_efficiency_page(step_grid, iqm_scores_by_alg, iqm_cis_by_alg, algorithms, xlabel, ylabel, pdf, title):
    fig, ax = plt.subplots(figsize=(9.6, 5.8), constrained_layout=True)
    for alg in algorithms:
        y = iqm_scores_by_alg[alg]
        ci = _ci_rows_first(iqm_cis_by_alg[alg])  # from your earlier patch
        ax.plot(step_grid, y, label=alg)
        ax.fill_between(step_grid, ci[:, 0], ci[:, 1], alpha=0.2, linewidth=0)

    # Avoid “1e8” glued to the axis label
    ax.ticklabel_format(useOffset=False, axis="x")
    fmt = ScalarFormatter(useMathText=True)
    fmt.set_powerlimits((-3, 3))
    ax.xaxis.set_major_formatter(fmt)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(title="Algorithm", loc="upper left", bbox_to_anchor=(1.02, 1.0))
    page_title(fig, title)
    pdf.savefig(fig)
    plt.close(fig)

def plot_prob_improvement_page(table: pd.DataFrame, pdf, title):
    fig, ax = plt.subplots(figsize=(9.2, 9.2), constrained_layout=True)
    vals = table.values.astype(float)
    im = ax.imshow(vals, vmin=0, vmax=1, aspect="equal")
    ax.set_xticks(range(len(table.columns)))
    ax.set_xticklabels(table.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(table.index)))
    ax.set_yticklabels(table.index)
    ax.set_title("Average probability of improvement (row beats column)")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel("Probability", rotation=270, labelpad=15)
    page_title(fig, title)
    pdf.savefig(fig)
    plt.close(fig)

# -------------------------- Reporting Pipeline ------------------------------

def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)

def save_aggregate_csv(out_dir: Path, aggregate_scores, aggregate_cis, metric_names, algorithms):
    ensure_dir(out_dir)
    rows = []
    for alg in algorithms:
        s = aggregate_scores[alg]
        c = _ci_rows_first(aggregate_cis[alg])   # <-- normalize
        for i, m in enumerate(metric_names):
            rows.append({
                "algorithm": alg,
                "metric": m,
                "estimate": float(s[i]),
                "ci_low": float(c[i, 0]),
                "ci_high": float(c[i, 1]),
            })
    pd.DataFrame(rows).to_csv(out_dir / "aggregate_summary.csv", index=False)

def save_steps_iqm_csv(out_dir: Path, step_grid, iqm_scores_by_alg, iqm_cis_by_alg):
    ensure_dir(out_dir)
    rows = []
    for alg, y in iqm_scores_by_alg.items():
        ci = _ci_rows_first(iqm_cis_by_alg[alg])
        for s, est, (lo, hi) in zip(step_grid, y, ci):
            rows.append({"algorithm": alg, "step": s, "iqm": est, "ci_low": lo, "ci_high": hi})
    pd.DataFrame(rows).to_csv(out_dir / "steps_iqm.csv", index=False)

def save_prob_improvement_csv(out_dir: Path, table: pd.DataFrame):
    ensure_dir(out_dir)
    table.to_csv(out_dir / "prob_improvement.csv")

def generate_report_for_metric(category: str, metric_name: str, grouped: Dict[str, Dict], out_root: Path,
                               reps: int, ci_level: float, max_steps: int, dpi: int, style: str):
    """
    grouped: dict alg_label -> {"by_env": env -> [RunSeries]}
    """
    set_style(style)
    metric_dir = safe_metric_name(metric_name)
    out_dir = out_root / category / metric_dir
    ensure_dir(out_dir)
    pdf_path = out_dir / "report.pdf"

    # 1) Build final-values matrices per algorithm (runs x envs)
    scores_dict = {}
    for alg, d in grouped.items():
        mat, envs = final_values_matrix(d["by_env"])
        # Filter by min runs later as needed
        scores_dict[alg] = mat

    # Filter algorithms with too few runs
    min_runs = 1
    scores_dict = {a: X for a, X in scores_dict.items() if X.shape[0] >= min_runs and X.shape[1] > 0}
    if not scores_dict:
        return

    algorithms = sorted(scores_dict.keys())

    # 2) Aggregate interval estimates (Median/IQM/Mean/OptGap)
    aggregate_scores, aggregate_cis, metric_names = compute_aggregate_estimates(scores_dict, reps=reps)

    # 3) Performance profiles
    perf_prof, perf_prof_cis, thresholds = compute_performance_profiles(scores_dict)

    # 4) Sample-efficiency curves (IQM over steps) per algorithm
    #    Interpolate each run to a shared grid per algorithm; then compute IQM over envs & runs.
    steps_grid_by_alg = {}
    interp_runs_by_alg = {}
    for alg, d in grouped.items():
        step_grid, runs_interp = standardize_step_grid(d["by_env"], max_steps=max_steps)
        if step_grid is None:
            continue
        steps_grid_by_alg[alg] = step_grid
        interp_runs_by_alg[alg] = runs_interp

    iqm_scores_by_alg = {}
    iqm_cis_by_alg = {}
    # Use the smallest common step grid across algorithms to make one page
    common_grid = None
    if steps_grid_by_alg:
        # pick shortest grid length across algorithms to ensure align
        common_len = min(len(g) for g in steps_grid_by_alg.values())
        # Uniformly downsample to common_len across all algs
        for alg in algorithms:
            if alg not in interp_runs_by_alg:
                continue
            # Downsample step grid and runs
            grid = steps_grid_by_alg[alg]
            idx = np.linspace(0, len(grid) - 1, num=common_len).astype(int)
            grid_ds = grid[idx]
            arr_runs_env_steps, _ = build_sample_efficiency_inputs(interp_runs_by_alg[alg])
            if arr_runs_env_steps.size == 0:
                continue
            arr_runs_env_steps = arr_runs_env_steps[:, :, idx]
            y, ci = compute_iqm_over_steps(arr_runs_env_steps, reps=reps)
            iqm_scores_by_alg[alg] = y
            iqm_cis_by_alg[alg] = ci
            common_grid = grid_ds

    # 5) Probability of improvement table
    prob_table, _ = prob_improvement_pairs(scores_dict, reps=reps)

    # ---------------- Write outputs ----------------
    # CSV summaries
    save_aggregate_csv(out_dir, aggregate_scores, aggregate_cis, metric_names, algorithms)
    if common_grid is not None and iqm_scores_by_alg:
        save_steps_iqm_csv(out_dir, common_grid, iqm_scores_by_alg, iqm_cis_by_alg)
    save_prob_improvement_csv(out_dir, prob_table)

    # PDF
    with PdfPages(pdf_path) as pdf:
        # Page 1: aggregate interval estimates
        plot_interval_estimates_page(
            aggregate_scores, aggregate_cis, metric_names, algorithms,
            xlabel=f"{metric_name}", pdf=pdf,
            title=f"{category.title()} — {metric_name} — Aggregate estimates"
        )
        # Page 2: performance profiles
        if perf_prof is not None:
            plot_performance_profiles_page(
                perf_prof, thresholds, perf_prof_cis, algorithms,
                xlabel=f"{metric_name} threshold (τ)", pdf=pdf,
                title=f"{category.title()} — {metric_name} — Performance profiles"
            )
        # Page 3: sample-efficiency curves (if available)
        if common_grid is not None and iqm_scores_by_alg:
            plot_sample_efficiency_page(
                common_grid, iqm_scores_by_alg, iqm_cis_by_alg, list(iqm_scores_by_alg.keys()),
                xlabel="Training step", ylabel=f"IQM({metric_name})", pdf=pdf,
                title=f"{category.title()} — {metric_name} — Sample efficiency (IQM over steps)"
            )
        # Page 4: probability of improvement heatmap
        if not prob_table.empty:
            plot_prob_improvement_page(prob_table, pdf,
                                       title=f"{category.title()} — {metric_name} — Probability of improvement")

    # Save a tiny README for the metric folder
    (out_dir / "README.txt").write_text(
        f"Report: {pdf_path.name}\n"
        f"Contains interval estimates (Median/IQM/Mean/OptGap), performance profiles, sample-efficiency curves,\n"
        f"and pairwise probability of improvement for algorithms (rows beat columns).\n"
    )

def main():
    args = parse_args()
    root = args.root
    out_root = args.out
    ensure_dir(out_root)

    data = collect_data(root)

    for category, metrics_dict in data.items():
        for metric_name, grouped_by_alg in tqdm(metrics_dict.items(),
                                               desc=f"Reporting: {category}", leave=False):
            # Skip obviously empty metrics
            nonempty = any(len(info["by_env"]) for info in grouped_by_alg.values())
            if not nonempty:
                continue
            generate_report_for_metric(
                category=category,
                metric_name=metric_name,
                grouped=grouped_by_alg,
                out_root=out_root,
                reps=args.reps,
                ci_level=args.ci,
                max_steps=args.max_steps,
                dpi=args.dpi,
                style=args.style
            )

if __name__ == "__main__":
    main()


