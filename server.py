"""Federated server: the training loop that ties data partitioning,
clients, the network simulator, DP accounting, aggregation, and
evaluation together into full communication rounds.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import List

import numpy as np

from fedpriv.data.synthetic import TabularDataset
from fedpriv.evaluation.metrics import evaluate
from fedpriv.federated.aggregation import fedavg_aggregate
from fedpriv.federated.client import FederatedClient
from fedpriv.federated.communication import NetworkSimulator
from fedpriv.models.mlp_numpy import NumpyMLP
from fedpriv.privacy.accountant import PrivacyAccountant
from fedpriv.privacy.mechanisms import MECHANISMS
from fedpriv.privacy.semantic_clipping import composed_sensitivity, default_group_clip_norms
from fedpriv.utils.config import ExperimentConfig
from fedpriv.utils.logger import MetricsWriter, get_logger


class FederatedServer:
    def __init__(self, cfg: ExperimentConfig, dataset: TabularDataset,
                 client_indices: List[np.ndarray]):
        self.cfg = cfg
        self.dataset = dataset
        self.rng = np.random.default_rng(cfg.seed)

        # Deliberately small hidden layer: client-level DP noise is added to
        # every one of the model's d parameters, so total noise power grows
        # with d while signal is capped by clip_norm regardless of d. Smaller
        # models -> better privacy-utility tradeoff for a fixed epsilon; this
        # is itself a real, citable finding (see PAPER_GUIDE.md).
        self.model = NumpyMLP(dataset.n_features, dataset.n_classes, hidden_dim=8, seed=cfg.seed)
        self.global_params = self.model.get_params_vector()

        self.clients = [
            FederatedClient(i, dataset.X_train[idx], dataset.y_train[idx], cfg)
            for i, idx in enumerate(client_indices)
        ]
        self.network = NetworkSimulator(cfg.network, seed=cfg.seed)

        # Central-DP-FedAvg (McMahan et al., 2018): the server aggregates
        # per-client-clipped updates with a FIXED denominator (the EXPECTED
        # cohort size, not the realized one — see aggregation.py), so a
        # single client changing their data can move the aggregate by at
        # most clip_norm / cohort_size. That quantity — not the raw
        # clip_norm — is the true sensitivity the accountant must be
        # calibrated against.
        self.cohort_size = max(1, round(cfg.federated.client_fraction * cfg.federated.n_clients))

        # With clip_mode="semantic", the effective total clip norm is the
        # Pythagorean sum of the per-group bounds (see semantic_clipping.py)
        # — equal to cfg.privacy.clip_norm under default uniform weighting,
        # but computed explicitly here so custom class_weights stay correct.
        if cfg.privacy.clip_mode == "semantic":
            self.semantic_group_norms = default_group_clip_norms(
                self.model.shapes, cfg.privacy.clip_norm, cfg.privacy.semantic_class_weights)
            effective_clip_norm = composed_sensitivity(self.semantic_group_norms)
        else:
            self.semantic_group_norms = None
            effective_clip_norm = cfg.privacy.clip_norm

        # Amplification by subsampling is opt-in (see accountant.py); off
        # by default, sampling_rate=1.0 reproduces the old behavior exactly.
        sampling_rate = (cfg.federated.client_fraction
                          if cfg.privacy.use_subsampling_amplification else 1.0)

        central_sensitivity = effective_clip_norm / self.cohort_size
        self.accountant = PrivacyAccountant(
            mechanism=cfg.privacy.mechanism if cfg.privacy.enabled else "none",
            epsilon=cfg.privacy.epsilon,
            delta=cfg.privacy.delta,
            clip_norm=central_sensitivity,
            total_rounds=cfg.federated.n_rounds,
            accountant=cfg.privacy.accountant,
            sampling_rate=sampling_rate,
        )

        self.noise_placement = cfg.privacy.noise_placement
        priv_active = cfg.privacy.enabled and cfg.privacy.mechanism != "none"
        if priv_active and self.noise_placement == "distributed":
            # Every client releases independently at FULL (undivided-by-
            # cohort-size) sensitivity — this accountant is calibrated for
            # that per-client release, not the aggregate. See client.py.
            self.client_accountant = PrivacyAccountant(
                mechanism=cfg.privacy.mechanism,
                epsilon=cfg.privacy.epsilon,
                delta=cfg.privacy.delta,
                clip_norm=effective_clip_norm,
                total_rounds=cfg.federated.n_rounds,
                accountant=cfg.privacy.accountant,
                sampling_rate=sampling_rate,
            )
        else:
            self.client_accountant = None

        # Whichever accountant is actually calibrating the noise that gets
        # added this run — used for spend tracking/reporting below.
        self.active_accountant = self.client_accountant or self.accountant

        out_dir = Path(cfg.output_dir) / "results" / cfg.run_id
        out_dir.mkdir(parents=True, exist_ok=True)
        self.out_dir = out_dir
        self.logger = get_logger("fedpriv.server", Path(cfg.output_dir) / "logs", cfg.run_id)
        self.metrics_writer = MetricsWriter(out_dir / "metrics.jsonl")

    def _select_clients(self) -> List[int]:
        n_select = max(1, int(round(self.cfg.federated.client_fraction * len(self.clients))))
        return list(self.rng.choice(len(self.clients), size=n_select, replace=False))

    def run(self) -> dict:
        fed_cfg = self.cfg.federated
        history = []
        start_time = time.time()

        for round_idx in range(1, fed_cfg.n_rounds + 1):
            round_start = time.time()
            selected = self._select_clients()

            updates = []
            n_dropped = 0
            total_upload_bytes = 0
            total_latency_ms = 0.0
            local_losses = []

            client_noise_sigma = self.client_accountant.noise_param if self.client_accountant else None
            for client_id in selected:
                client = self.clients[client_id]
                result = client.local_train(self.global_params, self.model, noise_sigma=client_noise_sigma)
                if result.update is None:
                    continue

                tx = self.network.simulate_upload(len(result.update))
                total_upload_bytes += tx.payload_bytes
                total_latency_ms += tx.latency_ms + tx.transmission_time_ms

                if not tx.delivered:
                    n_dropped += 1
                    continue

                updates.append((result.update, result.n_samples))
                local_losses.append(result.local_loss)

            if updates:
                priv_enabled = self.cfg.privacy.enabled and self.cfg.privacy.mechanism != "none"
                agg_update = fedavg_aggregate(
                    updates,
                    uniform_weight=priv_enabled,
                    fixed_denominator=self.cohort_size if priv_enabled else None,
                )
                if priv_enabled and self.noise_placement == "central":
                    # ONE noise draw added to the aggregate — central DP.
                    mech_fn = MECHANISMS[self.cfg.privacy.mechanism]
                    agg_update = mech_fn(agg_update, self.accountant.noise_param, self.rng)
                # else: noise_placement == "distributed" — each client
                # already added its own noise draw in local_train(), so
                # there is nothing left to add here.
                self.global_params = self.global_params + agg_update
            else:
                self.logger.warning(f"Round {round_idx}: all selected clients dropped, skipping update.")

            self.model.set_params_vector(self.global_params)
            val_metrics = evaluate(self.model, self.dataset.X_val, self.dataset.y_val)
            spend = self.active_accountant.spend_at_round(round_idx)

            round_time = time.time() - round_start
            participation_rate = len(updates) / max(1, len(selected))

            record = {
                "round": round_idx,
                "val_accuracy": val_metrics.accuracy,
                "val_loss": val_metrics.loss,
                "val_f1": val_metrics.f1,
                "val_precision": val_metrics.precision,
                "val_recall": val_metrics.recall,
                "mean_local_loss": float(np.mean(local_losses)) if local_losses else float("nan"),
                "n_selected": len(selected),
                "n_delivered": len(updates),
                "n_dropped": n_dropped,
                "participation_rate": participation_rate,
                "comm_bytes_round": total_upload_bytes,
                "comm_latency_ms_round": total_latency_ms,
                "round_time_sec": round_time,
                "privacy_epsilon_spent": spend.epsilon,
                "privacy_delta_spent": spend.delta,
                "noise_param": spend.sigma_or_scale,
            }
            history.append(record)
            self.metrics_writer.log(**record)

            if round_idx % max(1, fed_cfg.n_rounds // 10) == 0 or round_idx == fed_cfg.n_rounds:
                self.logger.info(
                    f"Round {round_idx:3d}/{fed_cfg.n_rounds} | "
                    f"val_acc={val_metrics.accuracy:.4f} val_loss={val_metrics.loss:.4f} | "
                    f"participation={participation_rate:.2f} | "
                    f"eps_spent={spend.epsilon:.3f}"
                )

        total_time = time.time() - start_time
        test_metrics = evaluate(self.model, self.dataset.X_test, self.dataset.y_test)

        summary = {
            "run_id": self.cfg.run_id,
            "dataset": self.dataset.name,
            "partition_strategy": self.cfg.partition.strategy,
            "privacy_mechanism": self.accountant.mechanism,
            "epsilon": self.cfg.privacy.epsilon,
            "delta": self.cfg.privacy.delta,
            "accountant": self.cfg.privacy.accountant,
            "clip_mode": self.cfg.privacy.clip_mode,
            "noise_placement": self.cfg.privacy.noise_placement,
            "use_subsampling_amplification": self.cfg.privacy.use_subsampling_amplification,
            "n_clients": fed_cfg.n_clients,
            "n_rounds": fed_cfg.n_rounds,
            "final_noise_param": self.active_accountant.noise_param,
            "test_accuracy": test_metrics.accuracy,
            "test_precision": test_metrics.precision,
            "test_recall": test_metrics.recall,
            "test_f1": test_metrics.f1,
            "test_loss": test_metrics.loss,
            "total_comm_bytes": sum(h["comm_bytes_round"] for h in history),
            "total_training_time_sec": total_time,
            "final_epsilon_spent": history[-1]["privacy_epsilon_spent"] if history else 0.0,
            "mean_participation_rate": float(np.mean([h["participation_rate"] for h in history])) if history else 0.0,
        }

        self.metrics_writer.close()
        return {"history": history, "summary": summary}

    def checkpoint(self, tag: str = "final") -> Path:
        ckpt_dir = Path(self.cfg.output_dir) / "checkpoints" / self.cfg.run_id
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        path = ckpt_dir / f"{tag}.npy"
        np.save(path, self.global_params)
        return path
