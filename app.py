#!/usr/bin/env python3
"""Command-line interface.

Single run:
    python app.py run --config configs/default.yaml \\
        --set federated.n_rounds=50 privacy.epsilon=1.0

Full paper sweep (epsilon x mechanism x partition):
    python app.py sweep --config configs/default.yaml --sweep paper_main

List available sweeps:
    python app.py sweep --list
"""
from __future__ import annotations

import argparse
import sys

from fedpriv.experiments import sweeps
from fedpriv.experiments.runner import run_single, run_sweep
from fedpriv.utils.config import load_config


def main():
    parser = argparse.ArgumentParser(description="Privacy-Preserving Federated Learning framework")
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="Run a single federated training experiment")
    run_p.add_argument("--config", default="configs/default.yaml")
    run_p.add_argument("--set", nargs="*", default=[], help="dot-path overrides, e.g. privacy.epsilon=0.5")

    sweep_p = sub.add_parser("sweep", help="Run a predefined multi-run comparison sweep")
    sweep_p.add_argument("--config", default="configs/default.yaml")
    sweep_p.add_argument("--sweep", default="epsilon_sweep")
    sweep_p.add_argument("--list", action="store_true")
    sweep_p.add_argument("--set", nargs="*", default=[])

    args = parser.parse_args()

    if args.command == "run":
        cfg = load_config(args.config, args.set)
        print(f"Running single experiment: {cfg.run_id}")
        result = run_single(cfg)
        print("\n=== SUMMARY ===")
        for k, v in result["summary"].items():
            print(f"{k:28s}: {v}")

    elif args.command == "sweep":
        if args.list:
            print("Available sweeps:", ", ".join(sweeps.SWEEPS.keys()))
            sys.exit(0)
        cfg = load_config(args.config, args.set)
        if args.sweep not in sweeps.SWEEPS:
            print(f"Unknown sweep '{args.sweep}'. Options: {list(sweeps.SWEEPS)}")
            sys.exit(1)
        grid = sweeps.SWEEPS[args.sweep]
        print(f"Running sweep '{args.sweep}' with grid: {grid}")
        summary_df = run_sweep(cfg, grid, sweep_name=args.sweep)
        print("\n=== COMPARISON TABLE ===")
        print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
