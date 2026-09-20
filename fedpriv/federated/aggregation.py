"""Aggregation strategies. Currently FedAvg (McMahan et al., 2017);
FedProx (Li et al., 2020) reuses the same aggregation and instead adds a
proximal term at the CLIENT during local training (see federated/client.py) —
this matches how FedProx is actually specified in the paper.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np


def fedavg_aggregate(updates: List[Tuple[np.ndarray, int]], uniform_weight: bool = False,
                      fixed_denominator: int = None) -> np.ndarray:
    """Average of client parameter UPDATES (deltas).

    Two weighting modes:
    - Sample-size weighted (default, `uniform_weight=False`): the classic
      FedAvg weighting, statistically efficient, used whenever privacy is
      disabled or for non-private comparisons.
    - Uniform, fixed-denominator (`uniform_weight=True`): every delivered
      update gets weight `1/fixed_denominator` (typically the EXPECTED
      cohort size, not the realized one). This is what central-DP-FedAvg
      requires: the denominator must not depend on which clients actually
      showed up, or the sensitivity bound used to calibrate server-side
      noise (see federated/server.py) would silently be violated by
      dropout/subsampling variance. If a client drops, its slot simply
      contributes zero rather than being redistributed to the others.

    `updates`: list of (update_vector, n_samples) from clients that
    successfully delivered their update this round (post-dropout filtering).
    """
    if not updates:
        raise ValueError("No client updates to aggregate (all clients dropped this round?)")

    if uniform_weight:
        denom = fixed_denominator or len(updates)
        agg = np.zeros_like(updates[0][0])
        for update, _n in updates:
            agg += update / denom
        return agg

    total_samples = sum(n for _, n in updates)
    agg = np.zeros_like(updates[0][0])
    for update, n in updates:
        agg += (n / total_samples) * update
    return agg
