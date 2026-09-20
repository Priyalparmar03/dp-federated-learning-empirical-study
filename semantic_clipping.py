"""Semantic (per-group) clipping.

`clipping.py` clips the WHOLE flat parameter-update vector to a single
global L2 norm bound C. This module instead clips each output CLASS's
own slice of the update (its column of W2, plus its bias in b2) to its
own norm bound, plus one more group for the shared trunk (W1, b1).

Motivation
----------
Under class imbalance (see the Adult-dataset finding in PAPER_GUIDE.md:
at tight epsilon a globally-clipped, globally-noised model collapses
onto the majority class — recall 0.011 at eps=0.1), a single global clip
treats every class's contribution to the update as equally "expensive"
to protect, even though the majority class dominates the update's norm
and crowds out minority-class signal under one shared bound. Splitting
the clip budget by class lets you allocate headroom unevenly (protect
minority classes with a larger relative share) instead of accepting
whatever the majority class leaves over.

Fairness w.r.t. sensitivity accounting
---------------------------------------
The whole point of an ablation is that clip_mode=global and
clip_mode=semantic must be compared at the SAME nominal privacy cost.
Since the groups below exactly partition the flat parameter vector (no
parameter is in more than one group, every parameter is in exactly one
group), the overall L2 sensitivity of a semantically-clipped update is
the Pythagorean sum sqrt(sum(c_g^2)) of the per-group bounds. With
uniform weighting, `default_group_clip_norms` sets each c_g so that this
sum equals exactly `total_clip_norm` — i.e. equal-weight semantic
clipping has IDENTICAL composed sensitivity to a global clip of the same
`total_clip_norm`. Custom (non-uniform) weights still normalize to the
same total, so any accuracy difference you measure is attributable to
*how* the budget is shaped, not to a larger or smaller total budget.
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import numpy as np

from fedpriv.models.mlp_numpy import MLPShapes


def _param_offsets(shapes: MLPShapes) -> Dict[str, int]:
    offsets = {}
    cursor = 0
    for name, shape in shapes.shapes():
        offsets[name] = cursor
        cursor += int(np.prod(shape))
    return offsets


def get_semantic_groups(shapes: MLPShapes) -> Dict[str, np.ndarray]:
    """Return {group_name: flat-vector index array}. Groups exactly
    partition range(shapes.n_params()) — every index appears in exactly
    one group.
    """
    offsets = _param_offsets(shapes)
    sizes = {name: int(np.prod(shape)) for name, shape in shapes.shapes()}

    groups: Dict[str, np.ndarray] = {
        "shared": np.concatenate([
            np.arange(offsets["W1"], offsets["W1"] + sizes["W1"]),
            np.arange(offsets["b1"], offsets["b1"] + sizes["b1"]),
        ])
    }

    hidden_dim, n_classes = shapes.hidden_dim, shapes.n_classes
    w2_start, b2_start = offsets["W2"], offsets["b2"]
    for g in range(n_classes):
        # W2 has shape (hidden_dim, n_classes) and is flattened row-major
        # (.ravel() default order 'C'), so column g's hidden_dim entries
        # sit at w2_start + r*n_classes + g for r in range(hidden_dim).
        col_idx = w2_start + np.arange(hidden_dim) * n_classes + g
        groups[f"class_{g}"] = np.concatenate([col_idx, [b2_start + g]])

    return groups


def default_group_clip_norms(shapes: MLPShapes, total_clip_norm: float,
                              class_weights: Optional[Dict] = None) -> Dict[str, float]:
    """Allocate `total_clip_norm` across the (n_classes + 1) semantic
    groups such that sqrt(sum(c_g^2)) == total_clip_norm exactly.

    `class_weights`: optional dict keyed by class index (int) and/or the
    string "shared", e.g. {0: 2.0} gives class 0 twice the RELATIVE
    weight of the other (default-weight-1.0) groups. The normalization
    below keeps the TOTAL composed sensitivity fixed at `total_clip_norm`
    regardless of how the weights are set, so reweighting only changes
    the *shape* of the allocation, never the overall privacy cost.
    """
    weights: Dict[str, float] = {f"class_{g}": 1.0 for g in range(shapes.n_classes)}
    weights["shared"] = 1.0
    if class_weights:
        for k, w in class_weights.items():
            key = f"class_{k}" if isinstance(k, int) else str(k)
            if key in weights:
                weights[key] = float(w)

    norm_factor = math.sqrt(sum(w ** 2 for w in weights.values()))
    return {name: total_clip_norm * w / norm_factor for name, w in weights.items()}


def clip_update_semantic(update: np.ndarray, shapes: MLPShapes,
                          group_clip_norms: Dict[str, float]) -> Tuple[np.ndarray, Dict[str, float]]:
    """Clip each semantic group of `update` independently to its own L2
    norm bound. Returns (clipped_update, {group_name: raw_pre_clip_norm}).
    """
    groups = get_semantic_groups(shapes)
    clipped = update.copy()
    raw_norms: Dict[str, float] = {}
    for name, idx in groups.items():
        c = group_clip_norms[name]
        sub = update[idx]
        norm = float(np.linalg.norm(sub))
        raw_norms[name] = norm
        if norm > c and norm > 0.0:
            clipped[idx] = sub * (c / norm)
    return clipped, raw_norms


def composed_sensitivity(group_clip_norms: Dict[str, float]) -> float:
    """Overall L2 sensitivity of a semantically-clipped update: the
    Pythagorean sum of the per-group clip norms (valid because the
    groups partition the vector — see module docstring)."""
    return math.sqrt(sum(c ** 2 for c in group_clip_norms.values()))
