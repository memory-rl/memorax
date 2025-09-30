import os
import glob
from pathlib import Path
import numpy as np
import pandas as pd
from collections import defaultdict

import matplotlib.pyplot as plt
from rliable import library as rly
from rliable import metrics as rlm
from rliable import plot_utils


def _read_series(csv_path: str) -> np.ndarray:
    df = pd.read_csv(csv_path)
    col = df.columns[-1]
    return df[col].to_numpy(dtype=float)


def load_scores(
    root: str = "logs",
    metric: str = "evaluation/episodic_returns.csv",
):
    root = Path(root)

    pattern = str(root / "*" / "**" / metric)
    files = glob.glob(pattern, recursive=True)

    nested = defaultdict(
        lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    )

    def _parse_metric_path(p: Path):
        parts = p.relative_to(root).parts
        seed = parts[-3]
        timestamp = parts[-4]
        network = parts[-5]
        algorithm = parts[-6]
        environment = parts[:-6]
        return environment, algorithm, network, timestamp, seed

    for f in files:
        p = Path(f)
        try:
            environment, algoritm, network, timestamp, seed = _parse_metric_path(p)
            series = _read_series(f)
            nested[environment][algoritm][network][timestamp][seed] = series
        except Exception as e:
            continue

    metrics = {}
    for env_key, algos in nested.items():
        for algo, nets in algos.items():
            for net, ts_map in nets.items():
                latest_ts = max(ts_map.keys())
                seed_map = ts_map[latest_ts]
                for seed, series in seed_map.items():
                    metrics.setdefault(env_key, {}).setdefault(algo, {}).setdefault(
                        net, {}
                    )[seed] = series

    return metrics


def save_fig(fig, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def iqm_over_time(x):
    return np.array([rlm.aggregate_iqm(x[..., t]) for t in range(x.shape[-1])])


def mean_over_time(x):
    return np.array([rlm.aggregate_mean(x[..., t]) for t in range(x.shape[-1])])


def median_over_time(x):
    return np.array([rlm.aggregate_median(x[..., t]) for t in range(x.shape[-1])])


def plot_sample_efficiency(environment, algorithm, networks, scores):

    # steps = np.arange(scores[sorted(networks)[0]].shape[-1])
    length = min(cube.shape[-1] for cube in scores.values())
    steps = np.arange(length)
    scores = {net: cube[..., :length] for net, cube in scores.items()}

    iqm_scores, iqm_cis = rly.get_interval_estimates(
        scores,
        iqm_over_time,
        reps=50_000,
    )

    ax = plot_utils.plot_sample_efficiency_curve(
        steps,
        iqm_scores,
        iqm_cis,
        algorithms=scores.keys(),
        xlabel="Environment Steps",
        ylabel="IQM Episodic Return",
        legend=True,
    )
    save_fig(ax.figure, f"plots/{environment}/{algorithm}/sample_efficiency_iqm.png")

    median_scores, median_cis = rly.get_interval_estimates(
        scores,
        median_over_time,
        reps=50_000,
    )
    ax = plot_utils.plot_sample_efficiency_curve(
        steps,
        median_scores,
        median_cis,
        algorithms=scores.keys(),
        xlabel="Environment Steps",
        ylabel="Median Episodic Return",
        legend=True,
    )
    save_fig(ax.figure, f"plots/{environment}/{algorithm}/sample_efficiency_median.png")

    mean_scores, mean_cis = rly.get_interval_estimates(
        scores,
        mean_over_time,
        reps=50_000,
    )

    ax = plot_utils.plot_sample_efficiency_curve(
        steps,
        mean_scores,
        mean_cis,
        algorithms=scores.keys(),
        xlabel="Environment Steps",
        ylabel="Mean Episodic Return",
        legend=True,
    )
    save_fig(ax.figure, f"plots/{environment}/{algorithm}/sample_efficiency_mean.png")


import copy
import numpy as np


def graft_mlp_into_recurrent(
    metrics: dict,
    mapping: dict,
    src_network: str = "MLP",
    label_template: str = "MLP",
    in_place: bool = False,
):
    """
    Copy the MLP network from each source algo into the target recurrent algo
    as a new 'network' entry. Works across all environments present in `metrics`.

    mapping: {"DQN": "DRQN", "PPO": "RPPO", "PQN": "PRQN"}
    """
    if not in_place:
        metrics = copy.deepcopy(metrics)

    for env_key, algos in metrics.items():
        for src_algo, tgt_algo in mapping.items():
            if src_algo not in algos:
                continue
            if src_network not in algos[src_algo]:
                continue

            src_seedmap = algos[src_algo][src_network]  # {seed: np.ndarray}
            if not src_seedmap:
                continue

            # ensure target algo exists
            tgt = algos.setdefault(tgt_algo, {})

            # decide a non-colliding network label inside target algo
            target_label = label_template.format(src=src_algo)
            candidate = target_label
            i = 2
            while candidate in tgt:
                candidate = f"{target_label} #{i}"
                i += 1

            # deep copy series to be safe
            tgt[candidate] = {
                seed: np.copy(np.asarray(series, dtype=float))
                for seed, series in src_seedmap.items()
            }

    return metrics


def plot_final_performance_vs_length(
    metrics,
    algorithm: str,
    env_base="MemoryChain-bsuite",
    lengths=(16, 32, 64, 128, 256, 512),
    aggregate="iqm",
    reps=50_000,
):
    agg_map = {
        "iqm": (rlm.aggregate_iqm, "IQM"),
        "median": (rlm.aggregate_median, "Median"),
        "mean": (rlm.aggregate_mean, "Mean"),
    }
    agg_fn, agg_name = agg_map[aggregate]

    def _parse_env_key(k):
        if isinstance(k, tuple):
            base = k[0]
            length = k[1] if len(k) > 1 else None
        else:
            parts = str(k).replace("\\", "/").split("/")
            base = parts[0] if parts else None
            length = parts[1] if len(parts) > 1 else None
        return base, length

    # Map (base, length_str) -> original key
    env_index = {}
    for k in metrics.keys():
        base, length = _parse_env_key(k)
        if base and length:
            env_index[(base, str(length))] = k

    # All networks observed for this algorithm across requested lengths
    nets = sorted(
        {
            net
            for L in lengths
            for key in [env_index.get((env_base, str(L)))]
            if key is not None and algorithm in metrics[key]
            for net in metrics[key][algorithm].keys()
        }
    )

    per_net = {n: {"x": [], "y": [], "lo": [], "hi": []} for n in nets}

    # rliable aggregator expects (runs, tasks, time); we’ll pass a single step
    def _final_scalar(x):
        return agg_fn(x[..., -1].ravel())

    def _to_scalar(z):
        z = np.asarray(z)
        return float(z.reshape(-1)[0])

    for L in lengths:
        key = env_index.get((env_base, str(L)))
        if key is None or algorithm not in metrics[key]:
            continue

        scores_at_L = {}
        for net in nets:
            if net not in metrics[key][algorithm]:
                continue
            last_vals = []
            for _seed, series in metrics[key][algorithm][net].items():
                arr = np.asarray(series, dtype=float)
                if arr.size == 0:
                    continue
                last_vals.append(arr[-1])  # final return
            if last_vals:
                scores_at_L[net] = np.asarray(last_vals, dtype=float)[:, None, None]

        if not scores_at_L:
            continue

        est, cis = rly.get_interval_estimates(scores_at_L, _final_scalar, reps=reps)
        for net in scores_at_L.keys():
            lo, hi = cis[net]
            per_net[net]["x"].append(L)
            per_net[net]["y"].append(_to_scalar(est[net]))
            per_net[net]["lo"].append(_to_scalar(lo))
            per_net[net]["hi"].append(_to_scalar(hi))

    # Keep only lengths common to all plotted networks
    sets = [set(v["x"]) for v in per_net.values() if v["x"]]
    common = sorted(set.intersection(*sets)) if sets else []
    if not common:
        raise RuntimeError("No common lengths across networks to plot.")

    steps = np.array(common)
    scores_dict, cis_dict = {}, {}
    for net, data in per_net.items():
        if not data["x"]:
            continue
        idx = [i for i, L in enumerate(data["x"]) if L in common]
        if not idx:
            continue
        xs = np.array([data["x"][i] for i in idx])
        o = np.argsort(xs)
        ys = np.array([data["y"][i] for i in idx])[o]
        los = np.array([data["lo"][i] for i in idx])[o]
        his = np.array([data["hi"][i] for i in idx])[o]
        scores_dict[net] = ys
        cis_dict[net] = (los, his)

    ax = plot_utils.plot_sample_efficiency_curve(
        steps,
        scores_dict,
        cis_dict,
        algorithms=list(scores_dict.keys()),  # labels = networks
        xlabel="Sequence Length",
        ylabel=f"{agg_name} Final Episodic Return",
        legend=True,
    )
    ax.set_title(f"{env_base} / {algorithm}: Final Performance vs Length")
    save_fig(
        ax.figure,
        f"plots/{env_base}/{algorithm}/final_performance_vs_length_{aggregate}.png",
    )


if __name__ == "__main__":
    metrics = load_scores(root="logs", metric="evaluation/episodic_returns.csv")

    mapping = {"DQN": "DRQN", "PPO": "RPPO", "PQN": "PRQN"}
    mapping = {k.lower(): v.lower() for k, v in mapping.items()}
    metrics = graft_mlp_into_recurrent(metrics, mapping, src_network="MLP")

    for environment, algorithms in metrics.items():
        for algorithm, networks in algorithms.items():
            scores = {}
            for network, seeds in networks.items():
                cube = np.stack(
                    [networks[network][seed] for seed in seeds],
                )
                cube = cube[:, None, :]
                scores[network] = cube
            print("Plotting for", environment, algorithm)
            plot_sample_efficiency(environment, algorithm, networks, scores)
    env_base = "MemoryChain-bsuite"
    lengths = (16, 32, 64, 128, 256, 512, 1024)
    aggregate = "iqm"

    # Discover algorithms present for the chosen env_base/lengths
    def _parse_env_key(k):
        if isinstance(k, tuple):
            base = k[0]
            length = k[1] if len(k) > 1 else None
        else:
            parts = str(k).replace("\\", "/").split("/")
            base = parts[0] if parts else None
            length = parts[1] if len(parts) > 1 else None
        return base, length

    lengths_str = {str(L) for L in lengths}
    algorithms = sorted(
        {
            algo
            for k in metrics.keys()
            for algo in metrics[k].keys()
            if (
                _parse_env_key(k)[0] == env_base
                and str(_parse_env_key(k)[1]) in lengths_str
            )
        }
    )

    for algo in algorithms:
        print(f"Plotting final performance vs length for {env_base} / {algo}")
        plot_final_performance_vs_length(
            metrics,
            algorithm=algo,
            env_base=env_base,
            lengths=lengths,
            aggregate=aggregate,
        )
