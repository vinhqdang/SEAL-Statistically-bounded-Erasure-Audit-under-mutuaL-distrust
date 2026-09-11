"""Two SOTA-representative baselines SEAL is benchmarked against.

1. ``parameter_diff_statistic`` -- the parameter-inspection audit. Included
   to reproduce, empirically, the well-known negative result that
   parameter distance cannot certify unlearning (an honest retrain and a
   dishonest run can be numerically closer or farther apart than each
   other by chance): expected detection power ~ chance (AUC ~ 0.5), and by
   construction it never touches individual retained records, so its
   retained-set leakage is ~0.

2. ``unrestricted_behavioral_statistic`` -- an *unrestricted* behavioral /
   membership-inference audit in the style of Tang, Joshi & Kundu
   (arXiv:2606.14518): the auditor is not limited to its own forgotten
   records or a bounded, noised query channel, but samples directly from
   the actual retained population to maximize detection power. This is the
   high-power / high-leakage end of the tradeoff that SEAL's tunable
   channel B is designed to dominate at matched leakage.
"""
from __future__ import annotations

import numpy as np

from .flmodel import FederatedSoftmax, Params


def parameter_diff_statistic(before: Params, after: Params) -> float:
    dW = np.linalg.norm(after.W - before.W)
    db = np.linalg.norm(after.b - before.b)
    return float(np.sqrt(dW ** 2 + db ** 2))


def unrestricted_behavioral_statistic(
    model: FederatedSoftmax,
    retained_X: np.ndarray, retained_y: np.ndarray,
    before: Params, after: Params,
    n_query: int, seed: int,
) -> float:
    rng = np.random.default_rng(seed)
    n = len(retained_X)
    idx = rng.choice(n, size=min(n_query, n), replace=False)
    Xq, yq = retained_X[idx], retained_y[idx]
    return float(np.mean(model.per_sample_loss(Xq, yq, after) - model.per_sample_loss(Xq, yq, before)))
