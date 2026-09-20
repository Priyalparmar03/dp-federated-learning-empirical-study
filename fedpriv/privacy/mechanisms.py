"""Differential Privacy noise mechanisms.

Both mechanisms operate on an already L2-clipped update of sensitivity
`clip_norm` (see clipping.py) and are calibrated from (epsilon, delta)
via the accountant module, which is the ONLY place noise scale is
derived from privacy budget — mechanisms.py just adds noise of a given
scale, so it stays trivially unit-testable and reusable.
"""
from __future__ import annotations

import numpy as np


def gaussian_mechanism(update: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """Add i.i.d. N(0, sigma^2) noise to every coordinate.

    `sigma` is the noise multiplier already scaled by the clipping norm
    (i.e. std = sigma), as produced by PrivacyAccountant.gaussian_sigma().
    """
    noise = rng.normal(loc=0.0, scale=sigma, size=update.shape)
    return update + noise


def laplace_mechanism(update: np.ndarray, scale: float, rng: np.random.Generator) -> np.ndarray:
    """Add i.i.d. Laplace(0, scale) noise to every coordinate.

    `scale` = clip_norm / epsilon_per_round (pure epsilon-DP, no delta),
    as produced by PrivacyAccountant.laplace_scale().
    """
    noise = rng.laplace(loc=0.0, scale=scale, size=update.shape)
    return update + noise


MECHANISMS = {
    "gaussian": gaussian_mechanism,
    "laplace": laplace_mechanism,
}
