# """Publication-quality plots. Every function saves both .png (for
# slides/README) and .pdf (vector, for LaTeX papers) into `out_dir`.
# """
# from __future__ import annotations

# from pathlib import Path
# from typing import Dict, List, Optional

# import matplotlib
# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd

# matplotlib.rcParams.update({
#     "figure.dpi": 130,
#     "savefig.dpi": 300,
#     "font.size": 11,
#     "axes.spines.top": False,
#     "axes.spines.right": False,
#     "axes.grid": True,
#     "grid.alpha": 0.25,
#     "figure.autolayout": True,
# })


# def _save(fig, out_dir: Path, name: str):
#     out_dir.mkdir(parents=True, exist_ok=True)
#     fig.savefig(out_dir / f"{name}.png", bbox_inches="tight")
#     fig.savefig(out_dir / f"{name}.pdf", bbox_inches="tight")
#     plt.close(fig)


# def plot_metric_vs_round(history: pd.DataFrame, metric: str, out_dir: Path,
#                           name: str, ylabel: Optional[str] = None, title: Optional[str] = None):
#     fig, ax = plt.subplots(figsize=(6, 4))
#     ax.plot(history["round"], history[metric], linewidth=2)
#     ax.set_xlabel("Communication Round")
#     ax.set_ylabel(ylabel or metric)
#     ax.set_title(title or f"{metric} vs. Round")
#     _save(fig, out_dir, name)


# def plot_multi_run_metric_vs_round(runs: Dict[str, pd.DataFrame], metric: str, out_dir: Path,
#                                     name: str, ylabel: Optional[str] = None,
#                                     title: Optional[str] = None, label_prefix: str = ""):
#     """One line per run (e.g. one per epsilon value) on the same axes."""
#     fig, ax = plt.subplots(figsize=(6.5, 4.5))
#     for label, hist in runs.items():
#         ax.plot(hist["round"], hist[metric], linewidth=2, label=f"{label_prefix}{label}")
#     ax.set_xlabel("Communication Round")
#     ax.set_ylabel(ylabel or metric)
#     ax.set_title(title or f"{metric} vs. Round")
#     ax.legend(frameon=False, fontsize=9)
#     _save(fig, out_dir, name)


# def plot_privacy_utility_tradeoff(summary_df: pd.DataFrame, out_dir: Path, name: str = "privacy_vs_accuracy",
#                                    group_col: Optional[str] = None):
#     fig, ax = plt.subplots(figsize=(6, 4.5))
#     if group_col and group_col in summary_df.columns:
#         for g, sub in summary_df.groupby(group_col):
#             sub = sub.sort_values("epsilon")
#             ax.plot(sub["epsilon"], sub["test_accuracy"], marker="o", linewidth=2, label=str(g))
#         ax.legend(frameon=False, title=group_col)
#     else:
#         sub = summary_df.sort_values("epsilon")
#         ax.plot(sub["epsilon"], sub["test_accuracy"], marker="o", linewidth=2)
#     ax.set_xscale("log")
#     ax.set_xlabel(r"Privacy Budget $\varepsilon$ (log scale)")
#     ax.set_ylabel("Test Accuracy")
#     ax.set_title("Privacy-Utility Tradeoff")
#     _save(fig, out_dir, name)


# def plot_mechanism_comparison(summary_df: pd.DataFrame, out_dir: Path, name: str = "gaussian_vs_laplace"):
#     fig, ax = plt.subplots(figsize=(6.5, 4.5))
#     for mech, sub in summary_df.groupby("privacy_mechanism"):
#         sub = sub.sort_values("epsilon")
#         ax.plot(sub["epsilon"], sub["test_accuracy"], marker="o", linewidth=2, label=mech.capitalize())
#     ax.set_xscale("log")
#     ax.set_xlabel(r"Privacy Budget $\varepsilon$ (log scale)")
#     ax.set_ylabel("Test Accuracy")
#     ax.set_title("Gaussian vs. Laplace Mechanism")
#     ax.legend(frameon=False)
#     _save(fig, out_dir, name)


# def plot_iid_vs_noniid(summary_df: pd.DataFrame, out_dir: Path, name: str = "iid_vs_noniid"):
#     fig, ax = plt.subplots(figsize=(6.5, 4.5))
#     for strat, sub in summary_df.groupby("partition_strategy"):
#         sub = sub.sort_values("epsilon")
#         ax.plot(sub["epsilon"], sub["test_accuracy"], marker="o", linewidth=2, label=strat)
#     ax.set_xscale("log")
#     ax.set_xlabel(r"Privacy Budget $\varepsilon$ (log scale)")
#     ax.set_ylabel("Test Accuracy")
#     ax.set_title("Effect of Data Heterogeneity on Privacy-Utility Tradeoff")
#     ax.legend(frameon=False)
#     _save(fig, out_dir, name)


# def plot_communication_cost(summary_df: pd.DataFrame, out_dir: Path, name: str = "communication_cost"):
#     fig, ax = plt.subplots(figsize=(6, 4.5))
#     sub = summary_df.sort_values("n_clients")
#     ax.bar(sub["n_clients"].astype(str), sub["total_comm_bytes"] / 1e6)
#     ax.set_xlabel("Number of Clients")
#     ax.set_ylabel("Total Communication (MB)")
#     ax.set_title("Communication Cost vs. Client Count")
#     _save(fig, out_dir, name)


# def plot_client_label_heatmap(partition_stats: np.ndarray, out_dir: Path, name: str = "client_label_distribution"):
#     fig, ax = plt.subplots(figsize=(7, 5))
#     im = ax.imshow(partition_stats, aspect="auto", cmap="viridis")
#     ax.set_xlabel("Class Label")
#     ax.set_ylabel("Client ID")
#     ax.set_title("Per-Client Label Distribution")
#     fig.colorbar(im, ax=ax, label="# samples")
#     _save(fig, out_dir, name)


# def plot_confusion_matrix(cm: np.ndarray, out_dir: Path, name: str = "confusion_matrix",
#                            class_names: Optional[List[str]] = None):
#     fig, ax = plt.subplots(figsize=(5.5, 5))
#     im = ax.imshow(cm, cmap="Blues")
#     n = cm.shape[0]
#     ticks = class_names or [str(i) for i in range(n)]
#     ax.set_xticks(range(n)); ax.set_xticklabels(ticks, rotation=45, ha="right")
#     ax.set_yticks(range(n)); ax.set_yticklabels(ticks)
#     for i in range(n):
#         for j in range(n):
#             ax.text(j, i, cm[i, j], ha="center", va="center",
#                      color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=8)
#     ax.set_xlabel("Predicted"); ax.set_ylabel("True")
#     ax.set_title("Confusion Matrix")
#     fig.colorbar(im, ax=ax)
#     _save(fig, out_dir, name)


# def plot_training_time_comparison(summary_df: pd.DataFrame, out_dir: Path, name: str = "training_time"):
#     fig, ax = plt.subplots(figsize=(6.5, 4.5))
#     sub = summary_df.sort_values("epsilon")
#     ax.bar(sub["epsilon"].astype(str), sub["total_training_time_sec"])
#     ax.set_xlabel(r"Privacy Budget $\varepsilon$")
#     ax.set_ylabel("Total Training Time (s)")
#     ax.set_title("Training Time Across Privacy Budgets")
#     _save(fig, out_dir, name)




"""Publication-quality plots. Every function saves both .png (for
slides/README) and .pdf (vector, for LaTeX papers) into `out_dir`.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.rcParams.update({
    "figure.dpi": 130,
    "savefig.dpi": 300,
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "figure.autolayout": True,
})


def _save(fig, out_dir: Path, name: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{name}.png", bbox_inches="tight")
    fig.savefig(out_dir / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_metric_vs_round(history: pd.DataFrame, metric: str, out_dir: Path,
                          name: str, ylabel: Optional[str] = None, title: Optional[str] = None):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(history["round"], history[metric], linewidth=2)
    ax.set_xlabel("Communication Round")
    ax.set_ylabel(ylabel or metric)
    ax.set_title(title or f"{metric} vs. Round")
    _save(fig, out_dir, name)


def plot_multi_run_metric_vs_round(runs: Dict[str, pd.DataFrame], metric: str, out_dir: Path,
                                    name: str, ylabel: Optional[str] = None,
                                    title: Optional[str] = None, label_prefix: str = ""):
    """One line per run (e.g. one per epsilon value) on the same axes."""
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for label, hist in runs.items():
        ax.plot(hist["round"], hist[metric], linewidth=2, label=f"{label_prefix}{label}")
    ax.set_xlabel("Communication Round")
    ax.set_ylabel(ylabel or metric)
    ax.set_title(title or f"{metric} vs. Round")
    ax.legend(frameon=False, fontsize=9)
    _save(fig, out_dir, name)


def plot_privacy_utility_tradeoff(summary_df: pd.DataFrame, out_dir: Path, name: str = "privacy_vs_accuracy",
                                   group_col: Optional[str] = None):
    fig, ax = plt.subplots(figsize=(6, 4.5))
    if group_col and group_col in summary_df.columns:
        for g, sub in summary_df.groupby(group_col):
            sub = sub.sort_values("epsilon")
            ax.plot(sub["epsilon"], sub["test_accuracy"], marker="o", linewidth=2, label=str(g))
        ax.legend(frameon=False, title=group_col)
    else:
        sub = summary_df.sort_values("epsilon")
        ax.plot(sub["epsilon"], sub["test_accuracy"], marker="o", linewidth=2)
    ax.set_xscale("log")
    ax.set_xlabel(r"Privacy Budget $\varepsilon$ (log scale)")
    ax.set_ylabel("Test Accuracy")
    ax.set_title("Privacy-Utility Tradeoff")
    _save(fig, out_dir, name)


def plot_privacy_utility_tradeoff_with_error(summary_df: pd.DataFrame, out_dir: Path,
                                              name: str = "privacy_vs_accuracy",
                                              group_col: Optional[str] = None,
                                              metric: str = "test_accuracy"):
    """Same as plot_privacy_utility_tradeoff but draws mean +/- std error
    bars, for sweeps run with multiple seeds (see experiments/runner.py's
    aggregate_seeds). Expects `<metric>` (mean) and `<metric>_std` columns.
    `group_col` can be any categorical column that varies in the sweep —
    e.g. "accountant", "clip_mode", or "noise_placement" — not just
    accountant as in the non-error variant.
    """
    std_col = f"{metric}_std"
    fig, ax = plt.subplots(figsize=(6, 4.5))

    def _errorbar(sub, label=None):
        sub = sub.sort_values("epsilon")
        yerr = sub[std_col].fillna(0.0) if std_col in sub.columns else None
        ax.errorbar(sub["epsilon"], sub[metric], yerr=yerr, marker="o", linewidth=2,
                    capsize=3, label=label)

    if group_col and group_col in summary_df.columns:
        for g, sub in summary_df.groupby(group_col):
            _errorbar(sub, label=str(g))
        ax.legend(frameon=False, title=group_col)
    else:
        _errorbar(summary_df)

    ax.set_xscale("log")
    ax.set_xlabel(r"Privacy Budget $\varepsilon$ (log scale)")
    ax.set_ylabel(metric.replace("_", " ").title())
    n_seeds = int(summary_df["n_seeds"].iloc[0]) if "n_seeds" in summary_df.columns and len(summary_df) else None
    title = "Privacy-Utility Tradeoff"
    if n_seeds:
        title += f" (mean \u00b1 std, n={n_seeds} seeds)"
    ax.set_title(title)
    _save(fig, out_dir, name)


def plot_mechanism_comparison(summary_df: pd.DataFrame, out_dir: Path, name: str = "gaussian_vs_laplace"):
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for mech, sub in summary_df.groupby("privacy_mechanism"):
        sub = sub.sort_values("epsilon")
        ax.plot(sub["epsilon"], sub["test_accuracy"], marker="o", linewidth=2, label=mech.capitalize())
    ax.set_xscale("log")
    ax.set_xlabel(r"Privacy Budget $\varepsilon$ (log scale)")
    ax.set_ylabel("Test Accuracy")
    ax.set_title("Gaussian vs. Laplace Mechanism")
    ax.legend(frameon=False)
    _save(fig, out_dir, name)


def plot_iid_vs_noniid(summary_df: pd.DataFrame, out_dir: Path, name: str = "iid_vs_noniid"):
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for strat, sub in summary_df.groupby("partition_strategy"):
        sub = sub.sort_values("epsilon")
        ax.plot(sub["epsilon"], sub["test_accuracy"], marker="o", linewidth=2, label=strat)
    ax.set_xscale("log")
    ax.set_xlabel(r"Privacy Budget $\varepsilon$ (log scale)")
    ax.set_ylabel("Test Accuracy")
    ax.set_title("Effect of Data Heterogeneity on Privacy-Utility Tradeoff")
    ax.legend(frameon=False)
    _save(fig, out_dir, name)


def plot_communication_cost(summary_df: pd.DataFrame, out_dir: Path, name: str = "communication_cost"):
    fig, ax = plt.subplots(figsize=(6, 4.5))
    sub = summary_df.sort_values("n_clients")
    ax.bar(sub["n_clients"].astype(str), sub["total_comm_bytes"] / 1e6)
    ax.set_xlabel("Number of Clients")
    ax.set_ylabel("Total Communication (MB)")
    ax.set_title("Communication Cost vs. Client Count")
    _save(fig, out_dir, name)


def plot_client_label_heatmap(partition_stats: np.ndarray, out_dir: Path, name: str = "client_label_distribution"):
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(partition_stats, aspect="auto", cmap="viridis")
    ax.set_xlabel("Class Label")
    ax.set_ylabel("Client ID")
    ax.set_title("Per-Client Label Distribution")
    fig.colorbar(im, ax=ax, label="# samples")
    _save(fig, out_dir, name)


def plot_confusion_matrix(cm: np.ndarray, out_dir: Path, name: str = "confusion_matrix",
                           class_names: Optional[List[str]] = None):
    fig, ax = plt.subplots(figsize=(5.5, 5))
    im = ax.imshow(cm, cmap="Blues")
    n = cm.shape[0]
    ticks = class_names or [str(i) for i in range(n)]
    ax.set_xticks(range(n)); ax.set_xticklabels(ticks, rotation=45, ha="right")
    ax.set_yticks(range(n)); ax.set_yticklabels(ticks)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=8)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    fig.colorbar(im, ax=ax)
    _save(fig, out_dir, name)


def plot_training_time_comparison(summary_df: pd.DataFrame, out_dir: Path, name: str = "training_time"):
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    sub = summary_df.sort_values("epsilon")
    ax.bar(sub["epsilon"].astype(str), sub["total_training_time_sec"])
    ax.set_xlabel(r"Privacy Budget $\varepsilon$")
    ax.set_ylabel("Total Training Time (s)")
    ax.set_title("Training Time Across Privacy Budgets")
    _save(fig, out_dir, name)
