"""Evaluation metrics shared by SEAL and the baselines.

Two axes make up the Pareto tradeoff the experiments report:
  * detection power  -- can the audit statistic tell an honest unlearning
    run apart from a dishonest one (Monte-Carlo AUC over many replicate
    trials, i.e. the power of the *best possible* single threshold)?
  * retained-set leakage -- given exactly what the audit releases, how well
    can an attacker tell a retained (member) record from a same-distribution
    non-member (the standard loss-based membership-inference proxy)?
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score

from .flmodel import FederatedSoftmax, Params


def _safe_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    if len(set(labels.tolist())) < 2 or np.allclose(scores, scores[0]):
        return 0.5
    auc = roc_auc_score(labels, scores)
    return max(auc, 1 - auc)


def detection_power(honest_stats: np.ndarray, dishonest_stats: np.ndarray) -> float:
    """AUC of the audit statistic distinguishing honest vs. dishonest
    mechanism outputs across Monte-Carlo replicate trials."""
    labels = np.concatenate([np.zeros(len(honest_stats)), np.ones(len(dishonest_stats))])
    scores = np.concatenate([honest_stats, dishonest_stats])
    return _safe_auc(labels, scores)


def leakage_auc(
    model: FederatedSoftmax,
    member_X: np.ndarray, member_y: np.ndarray,
    nonmember_X: np.ndarray, nonmember_y: np.ndarray,
    before: Params, after: Params,
    sigma: float = 0.0, l_max: float = 8.0, seed: int = 0,
) -> float:
    """Empirical retained-set membership-inference leakage of releasing the
    (possibly noised) loss-gap statistic on `member_X` (in the retained set)
    vs. `nonmember_X` (a same-distribution, never-trained-on pool)."""
    rng = np.random.default_rng(seed)

    def released(X, y):
        gap = np.clip(model.per_sample_loss(X, y, after), 0, l_max) - np.clip(model.per_sample_loss(X, y, before), 0, l_max)
        if sigma > 0 and np.isfinite(sigma):
            gap = gap + rng.normal(scale=sigma, size=gap.shape)
        return gap

    m_scores = released(member_X, member_y)
    n_scores = released(nonmember_X, nonmember_y)
    labels = np.concatenate([np.ones(len(m_scores)), np.zeros(len(n_scores))])
    scores = np.concatenate([m_scores, n_scores])
    return _safe_auc(labels, scores)
