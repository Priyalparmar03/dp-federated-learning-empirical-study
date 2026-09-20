"""Experiment runner.

`run_single(cfg)` executes one full federated training run and returns
its history + summary. `run_sweep(base_cfg, grid)` runs many configs
(varying epsilon, mechanism, partition strategy, dataset, client count,
...), collects everything into one comparison DataFrame/CSV, and
generates the full publication figure set automatically.
"""
from __future__ import annotations

import copy
import itertools
import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from fedpriv.data.dataset_manager import load_dataset
from fedpriv.data.partitioning import partition_data, partition_stats
from fedpriv.federated.server import FederatedServer
from fedpriv.utils.config import ExperimentConfig, config_to_dict
from fedpriv.utils.seed import set_seed
from fedpriv.visualization import plots


def run_single(cfg: ExperimentConfig, save_plots: bool = True) -> Dict[str, Any]:
    set_seed(cfg.seed)

    dataset = load_dataset(cfg.dataset, seed=cfg.seed)
    client_indices = partition_data(
        dataset.y_train,
        n_clients=cfg.federated.n_clients,
        strategy=cfg.partition.strategy,
        seed=cfg.seed,
        alpha=cfg.partition.dirichlet_alpha,
        labels_per_client=cfg.partition.labels_per_client,
        sigma=cfg.partition.quantity_skew_sigma,
    )

    server = FederatedServer(cfg, dataset, client_indices)
    result = server.run()
    server.checkpoint("final")

    out_dir = Path(cfg.output_dir) / "results" / cfg.run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    history_df = pd.DataFrame(result["history"])
    history_df.to_csv(out_dir / "history.csv", index=False)
    with open(out_dir / "summary.json", "w") as fh:
        json.dump(result["summary"], fh, indent=2)
    with open(out_dir / "config.json", "w") as fh:
        json.dump(config_to_dict(cfg), fh, indent=2)

    if save_plots:
        plot_dir = Path(cfg.output_dir) / "plots" / cfg.run_id
        plots.plot_metric_vs_round(history_df, "val_accuracy", plot_dir, "accuracy_vs_rounds",
                                    ylabel="Validation Accuracy")
        plots.plot_metric_vs_round(history_df, "val_loss", plot_dir, "loss_vs_rounds",
                                    ylabel="Validation Loss")
        plots.plot_metric_vs_round(history_df, "privacy_epsilon_spent", plot_dir,
                                    "privacy_budget_vs_rounds", ylabel=r"Cumulative $\varepsilon$ spent")
        plots.plot_metric_vs_round(history_df, "comm_bytes_round", plot_dir,
                                    "communication_per_round", ylabel="Bytes / round")
        pstats = partition_stats(dataset.y_train, client_indices)
        plots.plot_client_label_heatmap(pstats, plot_dir)

    return result


def aggregate_seeds(summary_df: pd.DataFrame, group_keys: List[str]) -> pd.DataFrame:
    """Collapse a multi-seed sweep (one row per (config, seed) combo) into
    one row per unique config, with mean and std across seeds for every
    `test_*` metric (plus final_epsilon_spent). `group_keys` are the
    summary_df columns that vary across the sweep grid, EXCLUDING "seed"
    itself — e.g. ["epsilon"] or ["epsilon", "clip_mode"].

    `<metric>_mean` is renamed back to `<metric>` so every existing
    single-seed plotting function keeps working unmodified on the
    aggregated frame; `<metric>_std` columns are added alongside for
    error-bar plotting. A `n_seeds` column records how many seeds went
    into each row (sanity check that nothing silently dropped a run).
    """
    metric_cols = [c for c in summary_df.columns
                   if c.startswith("test_") or c == "final_epsilon_spent"]

    agg_spec = {c: ["mean", "std"] for c in metric_cols}
    grouped = summary_df.groupby(group_keys, as_index=False).agg(agg_spec)
    grouped.columns = ["_".join([p for p in col if p]) if isinstance(col, tuple) else col
                        for col in grouped.columns]
    grouped = grouped.rename(columns={f"{c}_mean": c for c in metric_cols})

    n_seeds = summary_df.groupby(group_keys).size().reset_index(name="n_seeds")
    grouped = grouped.merge(n_seeds, on=group_keys)

    # Carry through one representative value of any other column (e.g.
    # accountant, privacy_mechanism, clip_mode if it's constant within
    # the group) so downstream plotting can still branch on it.
    non_metric_cols = [c for c in summary_df.columns
                        if c not in metric_cols and c not in group_keys and c != "seed"]
    if non_metric_cols:
        first_vals = summary_df.groupby(group_keys, as_index=False)[non_metric_cols].first()
        grouped = grouped.merge(first_vals, on=group_keys)

    return grouped


def run_sweep(base_cfg: ExperimentConfig, grid: Dict[str, List[Any]],
              sweep_name: str = "sweep") -> pd.DataFrame:
    """`grid` maps dot-paths (e.g. "privacy.epsilon") to lists of values.
    Cartesian product of all lists is executed; results collected into one
    comparison table + the standard set of cross-run figures.

    If "seed" is one of the grid keys, the raw per-seed table is written
    to comparison_table.csv AND a seed-aggregated (mean/std) table is
    written to comparison_table_aggregated.csv; plots are generated from
    the aggregated table with error bars. The returned DataFrame is the
    aggregated one in that case, the raw one otherwise (unchanged from
    before for any sweep that doesn't vary "seed").
    """
    keys = list(grid.keys())
    value_lists = [grid[k] for k in keys]

    rows = []
    histories = {}

    for combo in itertools.product(*value_lists):
        cfg = copy.deepcopy(base_cfg)
        label_parts = []
        for key, val in zip(keys, combo):
            _set_dotpath(cfg, key.split("."), val)
            label_parts.append(f"{key.split('.')[-1]}={val}")
        cfg.run_id = f"{sweep_name}__" + "_".join(label_parts).replace(" ", "")

        result = run_single(cfg, save_plots=False)
        rows.append(result["summary"])
        histories[cfg.run_id] = pd.DataFrame(result["history"])

    summary_df = pd.DataFrame(rows)
    out_dir = Path(base_cfg.output_dir) / "results" / sweep_name
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(out_dir / "comparison_table.csv", index=False)

    plot_dir = Path(base_cfg.output_dir) / "plots" / sweep_name

    if "seed" in keys:
        group_keys = [k.split(".")[-1] for k in keys if k != "seed"]
        agg_df = aggregate_seeds(summary_df, group_keys)
        agg_df.to_csv(out_dir / "comparison_table_aggregated.csv", index=False)
        _generate_sweep_plots(agg_df, histories, plot_dir)
        return agg_df

    _generate_sweep_plots(summary_df, histories, plot_dir)
    return summary_df


def _set_dotpath(obj, path: List[str], value: Any) -> None:
    attr = path[0]
    if len(path) == 1:
        setattr(obj, attr, value)
    else:
        _set_dotpath(getattr(obj, attr), path[1:], value)


def _generate_sweep_plots(summary_df: pd.DataFrame, histories: Dict[str, pd.DataFrame], plot_dir: Path) -> None:
    has_std = "n_seeds" in summary_df.columns

    if "epsilon" in summary_df.columns and summary_df["epsilon"].nunique() > 1:
        group_col = None
        if "accountant" in summary_df.columns and summary_df["accountant"].nunique() > 1:
            group_col = "accountant"
        elif "clip_mode" in summary_df.columns and summary_df["clip_mode"].nunique() > 1:
            group_col = "clip_mode"
        elif "noise_placement" in summary_df.columns and summary_df["noise_placement"].nunique() > 1:
            group_col = "noise_placement"

        if has_std:
            plots.plot_privacy_utility_tradeoff_with_error(summary_df, plot_dir, group_col=group_col)
        else:
            plots.plot_privacy_utility_tradeoff(summary_df, plot_dir, group_col=group_col)

    if "privacy_mechanism" in summary_df.columns and summary_df["privacy_mechanism"].nunique() > 1:
        plots.plot_mechanism_comparison(summary_df, plot_dir)

    if "partition_strategy" in summary_df.columns and summary_df["partition_strategy"].nunique() > 1:
        plots.plot_iid_vs_noniid(summary_df, plot_dir)

    if "n_clients" in summary_df.columns and summary_df["n_clients"].nunique() > 1:
        plots.plot_communication_cost(summary_df, plot_dir)

    if "epsilon" in summary_df.columns and summary_df["epsilon"].nunique() > 1:
        plots.plot_training_time_comparison(summary_df, plot_dir)

    if len(histories) > 1 and "epsilon" in summary_df.columns:
        run_to_eps = dict(zip(summary_df["run_id"], summary_df["epsilon"]))
        by_eps = {run_to_eps[rid]: h for rid, h in histories.items() if rid in run_to_eps}
        by_eps = dict(sorted(by_eps.items()))
        plots.plot_multi_run_metric_vs_round(
            by_eps, "val_accuracy", plot_dir, "accuracy_vs_rounds_by_epsilon",
            ylabel="Validation Accuracy", label_prefix="ε=",
        )
