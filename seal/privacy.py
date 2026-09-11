"""Privacy accounting for the SEAL boundary-probe channel.

Threat model for this channel: each probe response released by the server
is a bounded score (a clipped per-sample cross-entropy loss, range
[0, L_max]). Swapping which retained record actually sits nearest a given
probe can change that one released value by at most L_max, so global
sensitivity is L_max per query (standard bounded-output-range sensitivity,
as in score/output-perturbation mechanisms). The Gaussian mechanism is
applied per probe and the B probes of one audit are composed with
zero-concentrated differential privacy (zCDP), then converted to an
(epsilon, delta) guarantee -- this is what lets an auditor dial "how much
retained-set privacy this audit run is allowed to cost" and get back the
noise level that enforces it.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class PrivacyBudget:
    epsilon: float
    delta: float
    n_queries: int
    sensitivity: float
    sigma: float
    rho_total: float

    def as_dict(self) -> dict:
        return {
            "epsilon": self.epsilon,
            "delta": self.delta,
            "n_queries": self.n_queries,
            "sensitivity": self.sensitivity,
            "sigma": self.sigma,
            "rho_total": self.rho_total,
        }


def rho_to_epsilon(rho: float, delta: float) -> float:
    """zCDP -> (epsilon, delta)-DP conversion (Bun & Steinke 2016, Prop. 3.3)."""
    if rho <= 0:
        return 0.0
    return rho + 2.0 * math.sqrt(rho * math.log(1.0 / delta))


def epsilon_to_rho(epsilon: float, delta: float) -> float:
    """Invert rho_to_epsilon for the smallest rho achieving `epsilon` at `delta`."""
    if epsilon <= 0:
        return 0.0
    c = math.sqrt(math.log(1.0 / delta))
    s = -c + math.sqrt(c * c + epsilon)
    return max(s, 0.0) ** 2


def calibrate_sigma(epsilon: float, delta: float, n_queries: int, sensitivity: float) -> PrivacyBudget:
    """Return the per-query Gaussian noise std that spends exactly
    (epsilon, delta) total privacy budget over `n_queries` composed probes."""
    if epsilon <= 0 or n_queries <= 0:
        return PrivacyBudget(epsilon=0.0, delta=delta, n_queries=n_queries, sensitivity=sensitivity, sigma=math.inf, rho_total=0.0)
    rho_total = epsilon_to_rho(epsilon, delta)
    rho_per_query = rho_total / n_queries
    sigma = sensitivity / math.sqrt(2.0 * rho_per_query) if rho_per_query > 0 else math.inf
    return PrivacyBudget(epsilon=epsilon, delta=delta, n_queries=n_queries, sensitivity=sensitivity, sigma=sigma, rho_total=rho_total)


def spent_epsilon(sigma: float, n_queries: int, sensitivity: float, delta: float) -> float:
    """The (epsilon, delta) actually spent by running `n_queries` Gaussian-mechanism
    probes at noise level `sigma`, for reporting / auditing the auditor itself."""
    if sigma == math.inf or n_queries == 0:
        return 0.0
    rho_per_query = (sensitivity ** 2) / (2.0 * sigma ** 2)
    return rho_to_epsilon(rho_per_query * n_queries, delta)
