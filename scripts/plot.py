import os
import glob
import numpy as np
import pandas as pd
from collections import defaultdict

import matplotlib.pyplot as plt
from rliable import library as rly
from rliable import metrics
from rliable import plot_utils



def _read_series(csv_path: str) -> np.ndarray:
    df = pd.read_csv(csv_path)
    col = df.columns[-1]
    return df[col].to_numpy(dtype=float)


def _load_metric(
    root: str,
    environment: str,
    algorithm: str,
    seed: str,
    metric="evaluation/episodic_returns.csv",
):
    paths = os.path.join(root, environment, algorithm, "*", "*", seed, metric)
    for path in glob.glob(paths):
        try:
            return _read_series(path)
        except FileNotFoundError:
            print("File not found:", path)
            pass
    raise FileNotFoundError(f"File not found: {paths}")


def load_scores(
    root: str = "logs",
    metric: str = "evaluation/episodic_returns.csv",
):
    environments = set()
    algorithms = set()
    seeds = set()

    for environment in os.listdir(root):
        if not os.path.isdir(os.path.join(root, environment)):
            continue
        environments.add(environment)
        for algorithm in os.listdir(os.path.join(root, environment)):
            if not os.path.isdir(os.path.join(root, environment, algorithm)):
                continue
            algorithms.add(algorithm)
            for network in os.listdir(os.path.join(root, environment, algorithm)):
                if not os.path.isdir(
                    os.path.join(root, environment, algorithm, network)
                ):
                    continue
                for timestamp in os.listdir(
                    os.path.join(root, environment, algorithm, network)
                ):
                    if not os.path.isdir(
                        os.path.join(root, environment, algorithm, network, timestamp)
                    ):
                        continue
                    for seed in os.listdir(
                        os.path.join(root, environment, algorithm, network, timestamp)
                    ):
                        if not os.path.isdir(
                            os.path.join(
                                root, environment, algorithm, network, timestamp, seed
                            )
                        ):
                            continue
                        if seed == "config.yaml":
                            continue
                        seeds.add(seed)

    environments = sorted(environments)
    algorithms = sorted(algorithms)
    seeds = sorted(seeds)

    metrics = defaultdict(dict)
    for algorithm in algorithms:
        for environment in environments:
            for seed in seeds:
                if environment not in metrics[algorithm]:
                    metrics[algorithm][environment] = {}
                metrics[algorithm][environment][seed] = _load_metric(
                    root, environment, algorithm, seed, metric=metric
                )

    scores = {}
    for algorithm in algorithms:

        cube = np.stack(
            [
                np.stack(
                    [
                        metrics[algorithm][environment][seed]
                        for environment in environments
                    ],
                    axis=0,
                )
                for seed in seeds
            ],
            axis=0,
        )
        scores[algorithm] = cube

    return algorithms, environments, scores, seeds


def iqm_over_time(x):
    return np.array([metrics.aggregate_iqm(x[..., t]) for t in range(x.shape[-1])])


def mean_over_time(x):
    return np.array([metrics.aggregate_mean(x[..., t]) for t in range(x.shape[-1])])


def median_over_time(x):
    return np.array([metrics.aggregate_median(x[..., t]) for t in range(x.shape[-1])])

def save_fig(fig, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path)
    plt.close(fig)

def aggregate(score):
    return np.array(
        [
            metrics.aggregate_median(score),
            metrics.aggregate_iqm(score),
            metrics.aggregate_mean(score),
            metrics.aggregate_optimality_gap(score),
        ]
    )

def plot_aggregates(algorithms, scores):
    final_scores = {a: v[..., -1] for a, v in scores.items()}

    aggregate_scores, aggregate_score_cis = rly.get_interval_estimates(
        final_scores, aggregate, reps=50_000
    )
    fig, ax = plot_utils.plot_interval_estimates(
        aggregate_scores,
        aggregate_score_cis,
        metric_names=["Median", "IQM", "Mean", "Optimality Gap"],
        algorithms=algorithms,
        xlabel="Return",
    )
    save_fig(fig, "plots/aggregate.png")



def plot_sample_efficiency(algorithms, scores):

    iqm_scores, iqm_cis = rly.get_interval_estimates(
        scores,
        iqm_over_time,
        reps=50_000,
    )
    median_scores, median_cis = rly.get_interval_estimates(
        scores,
        median_over_time,
        reps=50_000,
    )
    mean_scores, mean_cis = rly.get_interval_estimates(
        scores,
        mean_over_time,
        reps=50_000,
    )
    steps = np.arange(scores[algorithms[0]].shape[-1])  # 0..T-1

    ax = plot_utils.plot_sample_efficiency_curve(
        steps,
        iqm_scores,
        iqm_cis,
        algorithms=algorithms,
        xlabel="Environment Steps",
        ylabel="IQM Episodic Return",
        legend=True
    )
    save_fig(ax.figure, "plots/sample_efficiency_iqm.png")

    ax = plot_utils.plot_sample_efficiency_curve(
        steps,
        median_scores,
        median_cis,
        algorithms=algorithms,
        xlabel="Environment Steps",
        ylabel="Median Episodic Return",
        legend=True
    )
    save_fig(ax.figure, "plots/sample_efficiency_median.png")

    ax = plot_utils.plot_sample_efficiency_curve(
        steps,
        mean_scores,
        mean_cis,
        algorithms=algorithms,
        xlabel="Environment Steps",
        ylabel="Mean Episodic Return",
        legend=True
    )
    save_fig(ax.figure, "plots/sample_efficiency_mean.png")

if __name__ == "__main__":
    algorithms, tasks, scores, seeds = load_scores(
        root="logs", metric="evaluation/episodic_returns.csv"
    )
    plot_aggregates(algorithms, scores)
    plot_sample_efficiency(algorithms, scores)
