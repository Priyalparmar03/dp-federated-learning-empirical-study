"""L2 norm clipping — the sensitivity-bounding step every DP mechanism relies on.

We clip at the CLIENT-UPDATE level (i.e. the full parameter delta a client
sends after local training), which gives user-level / client-level DP —
the natural granularity in federated learning, as opposed to per-example
DP-SGD which bounds each individual training example's gradient. This is
documented explicitly because it is a real methodological choice a
reviewer will ask about (see PAPER_GUIDE.md, "Threat model" section).
"""
from __future__ import annotations

from typing import Tuple

import numpy as np


def clip_update(update: np.ndarray, clip_norm: float) -> Tuple[np.ndarray, float]:
    """Clip `update` to L2 norm <= clip_norm. Returns (clipped_update, original_norm)."""
    norm = float(np.linalg.norm(update))
    if norm <= clip_norm or norm == 0.0:
        return update, norm
    return update * (clip_norm / norm), norm
