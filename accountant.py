"""Privacy Accountant.

Turns a *target* (epsilon, delta) budget for the whole T-round training
run into a *per-round noise scale*, and — going the other way — reports
cumulative privacy spend at any round r <= T (used for the "privacy
budget vs. round" plot). This is the only place noise scale is derived
from (epsilon, delta); mechanisms.py just consumes the resulting sigma.

Two accounting modes are implemented on purpose, so the framework can
run an ablation on accountant tightness (a real methodological point,
see PAPER_GUIDE.md):

1. ``basic_composition`` — naive sequential composition. Splits the
   total budget evenly across rounds: eps_round = eps_total / T,
   delta_round = delta_total / T. Uses the classical Gaussian mechanism
   bound (Dwork & Roth, 2014, Thm A.1): sigma >= C * sqrt(2 ln(1.25/delta)) / eps.
   Simple, always valid, but pessimistic — it overestimates noise needed
   for a given end-to-end budget as T grows.

2. ``analytic_gaussian`` — zero-Concentrated-DP (zCDP) composition
   (Bun & Steinke, 2016). A sigma-Gaussian mechanism releasing a query of
   sensitivity C satisfies rho-zCDP with rho = C^2 / (2 sigma^2). zCDP
   composes ADDITIVELY over rounds (rho_total = sum(rho_i)), and any
   rho-zCDP mechanism is (rho + 2*sqrt(rho * ln(1/delta)), delta)-DP for
   every delta > 0. We invert this conversion to find the per-round rho
   (hence sigma) that hits the target end-to-end (epsilon, delta) after T
   rounds. This is substantially tighter than basic composition and is
   what modern DP-SGD/DP-FL libraries (e.g. Opacus) use under the hood
   (they use the more refined moments accountant / PRV accountant, which
   is even tighter still — noted as a limitation in PAPER_GUIDE.md).

Laplace uses pure epsilon-DP with basic (additive) composition, which is
exact for that mechanism — no approximation involved.

PRIVACY AMPLIFICATION BY SUBSAMPLING (optional, off by default)
-----------------------------------------------------------------
Each round only `sampling_rate` (= federated.client_fraction) of clients
participate. A mechanism applied to a randomly subsampled cohort leaks
strictly less than the same mechanism applied to the full population —
"amplification by subsampling". The exact amplified bound requires a
numeric RDP/PRV accountant (what Opacus/dp-accounting use under the
hood); we use the standard SMALL-q approximation instead, which is
simple, closed-form, and clearly documented as non-tight:

    noise_scale_amplified ≈ noise_scale_unamplified * sampling_rate

This is applied identically to both the Gaussian (analytic_gaussian
accountant) and Laplace mechanisms, and is exact for Laplace only in the
eps << 1 regime. It is intentionally NOT applied to `basic_composition`,
which exists specifically as the naive, pessimistic baseline in the
accountant-tightness ablation (see PAPER_GUIDE.md) — amplifying it too
would blur that comparison. `sampling_rate` defaults to 1.0 (no
amplification, i.e. exactly the old behavior) so existing results stay
reproducible unless `privacy.use_subsampling_amplification` is turned on
explicitly.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class PrivacySpend:
    epsilon: float
    delta: float
    sigma_or_scale: float
    mechanism: str
    round: int
    total_rounds: int


class PrivacyAccountant:
    def __init__(self, mechanism: str, epsilon: float, delta: float,
                 clip_norm: float, total_rounds: int, accountant: str = "analytic_gaussian",
                 sampling_rate: float = 1.0):
        self.mechanism = mechanism
        self.epsilon_total = epsilon
        self.delta_total = delta
        self.clip_norm = clip_norm
        self.total_rounds = max(1, total_rounds)
        self.accountant_type = accountant
        # Clamp to (0, 1]; 1.0 = no amplification (old default behavior).
        self.sampling_rate = max(min(sampling_rate, 1.0), 1e-6)

        if mechanism == "gaussian":
            self.noise_param = self._gaussian_sigma()
        elif mechanism == "laplace":
            self.noise_param = self._laplace_scale()
        elif mechanism == "none":
            self.noise_param = 0.0
        else:
            raise ValueError(f"Unknown mechanism '{mechanism}'")

    # ---------- Gaussian ----------

    def _gaussian_sigma(self) -> float:
        if self.accountant_type == "basic_composition":
            eps_round = self.epsilon_total / self.total_rounds
            delta_round = self.delta_total / self.total_rounds
            return self._classical_gaussian_sigma(eps_round, delta_round)
        elif self.accountant_type == "analytic_gaussian":
            rho_total = self._eps_delta_to_rho(self.epsilon_total, self.delta_total)
            rho_round = rho_total / self.total_rounds
            if rho_round <= 0:
                return float("inf")
            sigma = self.clip_norm / math.sqrt(2 * rho_round)
            return sigma * self._amplification_factor()
        else:
            raise ValueError(f"Unknown accountant '{self.accountant_type}'")

    def _amplification_factor(self) -> float:
        """Small-q approximation of privacy amplification by subsampling
        (see module docstring). Returns 1.0 (no-op) unless sampling_rate < 1.
        """
        return self.sampling_rate

    @staticmethod
    def _classical_gaussian_sigma(epsilon: float, delta: float) -> float:
        # Dwork & Roth (2014), Appendix A: valid for epsilon in (0, 1].
        # We still apply it outside that range as a standard, documented
        # approximation (common practice for pedagogical accountants).
        return math.sqrt(2.0 * math.log(1.25 / delta)) / max(epsilon, 1e-12)

    @staticmethod
    def _eps_delta_to_rho(epsilon: float, delta: float) -> float:
        """Invert eps = rho + 2*sqrt(rho * ln(1/delta)) for rho (Bun & Steinke, 2016)."""
        ln_inv_delta = math.log(1.0 / delta)
        u = -math.sqrt(ln_inv_delta) + math.sqrt(ln_inv_delta + epsilon)
        return max(u, 0.0) ** 2

    @staticmethod
    def _rho_to_eps(rho: float, delta: float) -> float:
        return rho + 2.0 * math.sqrt(rho * math.log(1.0 / delta))

    # ---------- Laplace ----------

    def _laplace_scale(self) -> float:
        eps_round = self.epsilon_total / self.total_rounds  # basic composition, exact for Laplace
        scale = self.clip_norm / max(eps_round, 1e-12)
        return scale * self._amplification_factor()

    # ---------- Reporting ----------

    def spend_at_round(self, r: int) -> PrivacySpend:
        """Cumulative (epsilon, delta) spent after `r` rounds of participation."""
        r = max(1, min(r, self.total_rounds))
        if self.mechanism == "gaussian":
            if self.accountant_type == "analytic_gaussian":
                rho_total = self._eps_delta_to_rho(self.epsilon_total, self.delta_total)
                rho_round = rho_total / self.total_rounds
                eps = self._rho_to_eps(rho_round * r, self.delta_total)
            else:
                eps = (self.epsilon_total / self.total_rounds) * r
            delta = self.delta_total
        elif self.mechanism == "laplace":
            eps = (self.epsilon_total / self.total_rounds) * r
            delta = 0.0
        else:
            eps, delta = 0.0, 0.0

        return PrivacySpend(
            epsilon=eps, delta=delta, sigma_or_scale=self.noise_param,
            mechanism=self.mechanism, round=r, total_rounds=self.total_rounds,
        )

    def final_spend(self) -> PrivacySpend:
        return self.spend_at_round(self.total_rounds)
