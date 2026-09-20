"""Federated client.

Each client: (1) receives the global parameter vector, (2) trains locally
for `local_epochs` epochs of mini-batch SGD (optionally with a FedProx
proximal term), (3) computes its update = local_params - global_params,
and (4) clips that update to the configured L2 norm bound C before
handing it to the network simulator for "transmission" back to the
server.

Clipping happens HERE (per client) so no single client can ever
contribute an unbounded update, either as one global L2 bound
(`privacy.clip_mode == "global"`, see privacy/clipping.py) or split per
output class plus a shared trunk group (`clip_mode == "semantic"`, see
privacy/semantic_clipping.py — the latter lets minority classes get a
larger relative clip budget instead of being crowded out by the
majority class under one shared bound).

DP NOISE placement is configurable via `privacy.noise_placement`:
- "central" (default): the SERVER adds ONE noise draw to the aggregate,
  once per round, after aggregation (see federated/server.py) — the
  standard "DP-FedAvg" construction of McMahan et al. (2018, "Learning
  Differentially Private Recurrent Language Models"). Sensitivity here
  is C / cohort_size, since a single client's change can move the
  fixed-denominator average by at most that much.
- "distributed": every CLIENT adds its own independent noise draw here,
  before transmission, calibrated to its own full (undivided)
  sensitivity C. Included as an ablation baseline (see PAPER_GUIDE.md
  6.1) — central placement's noise/cohort_size scaling is far more
  sample-efficient than every client paying full-sensitivity noise
  independently, so central is what real DP-FL systems deploy and what
  this framework defaults to.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from fedpriv.models.mlp_numpy import NumpyMLP
from fedpriv.privacy.clipping import clip_update
from fedpriv.privacy.mechanisms import MECHANISMS
from fedpriv.privacy.semantic_clipping import clip_update_semantic, default_group_clip_norms
from fedpriv.utils.config import ExperimentConfig


@dataclass
class ClientUpdateResult:
    client_id: int
    n_samples: int
    update: Optional[np.ndarray]   # None if dropped
    raw_update_norm: float
    local_loss: float
    delivered: bool


class FederatedClient:
    def __init__(self, client_id: int, X: np.ndarray, y: np.ndarray, cfg: ExperimentConfig):
        self.client_id = client_id
        self.X = X
        self.y = y
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed * 1000 + client_id)

    @property
    def n_samples(self) -> int:
        return len(self.y)

    def local_train(self, global_params: np.ndarray, model: NumpyMLP,
                     noise_sigma: Optional[float] = None) -> ClientUpdateResult:
        """`noise_sigma`: only used when `privacy.noise_placement ==
        "distributed"` — the server precomputes this from a second
        accountant calibrated to full per-client sensitivity and passes
        it down each round (see federated/server.py). Ignored (and
        should be None) under the default "central" placement.
        """
        fed_cfg = self.cfg.federated
        priv_cfg = self.cfg.privacy

        model.set_params_vector(global_params.copy())
        n = len(self.y)
        if n == 0:
            return ClientUpdateResult(self.client_id, 0, None, 0.0, float("nan"), False)

        idx_all = np.arange(n)
        last_loss = float("nan")
        for _epoch in range(fed_cfg.local_epochs):
            self.rng.shuffle(idx_all)
            for start in range(0, n, fed_cfg.batch_size):
                batch_idx = idx_all[start : start + fed_cfg.batch_size]
                Xb, yb = self.X[batch_idx], self.y[batch_idx]

                loss, grad = model.loss_and_grad_vector(Xb, yb)

                if fed_cfg.algorithm == "fedprox":
                    # proximal term pulls local params toward the global params
                    current = model.get_params_vector()
                    grad = grad + fed_cfg.fedprox_mu * (current - global_params)

                new_params = model.get_params_vector() - fed_cfg.learning_rate * grad
                model.set_params_vector(new_params)
                last_loss = loss

        local_params = model.get_params_vector()
        update = local_params - global_params

        raw_norm = float(np.linalg.norm(update))

        if priv_cfg.enabled and priv_cfg.mechanism != "none":
            if priv_cfg.clip_mode == "semantic":
                group_norms = default_group_clip_norms(
                    model.shapes, priv_cfg.clip_norm, priv_cfg.semantic_class_weights)
                update, _ = clip_update_semantic(update, model.shapes, group_norms)
            else:
                update, _ = clip_update(update, priv_cfg.clip_norm)

            if priv_cfg.noise_placement == "distributed" and noise_sigma is not None:
                mech_fn = MECHANISMS[priv_cfg.mechanism]
                update = mech_fn(update, noise_sigma, self.rng)

        return ClientUpdateResult(
            client_id=self.client_id,
            n_samples=n,
            update=update,
            raw_update_norm=raw_norm,
            local_loss=last_loss,
            delivered=True,  # dropout is decided by the network simulator, not here
        )
