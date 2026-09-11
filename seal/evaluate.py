"""Experiment orchestration: for one dataset, run many Monte-Carlo trials of
(honest mechanism, dishonest mechanism) pairs, audit each with SEAL (at a
sweep of epsilon budgets) and with the two baselines, and report detection
power vs. retained-set leakage.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .flmodel import FederatedSoftmax
from .data import Population, split_population, nearest_neighbor_pool
from .mechanisms import honest_finetune_unlearn, dishonest_lazy, dishonest_spoof
from .certificate import SealCertificate, build_calibration_replicates
from .baselines import parameter_diff_statistic, unrestricted_behavioral_statistic
from .metrics import detection_power, leakage_auc


@dataclass
class TrialConfig:
    k_clients: int = 5
    forget_client: int = 0
    forget_frac: float = 0.2
    calib_frac: float = 0.15
    holdout_frac: float = 0.15
    fl_rounds: int = 25
    local_epochs: int = 1
    lr: float = 0.3
    batch_size: int = 32
    unlearn_extra_rounds: int = 8
    spoof_epochs: int = 40
    spoof_lr: float = 0.5
    n_calib_reps: int = 25
    n_query_unrestricted: int = 60
    n_probes: int = 50
    jitter: float = 0.4
    l_max: float = 8.0
    beta_target: float = 0.05
    delta_audit: float = 1e-5
    n_neighbors_leak: int = 3


def _fit_base_model(pop: Population, split, cfg: TrialConfig, seed: int):
    model = FederatedSoftmax(pop.n_features, pop.n_classes, l2=1e-3)
    model.init(seed=seed)
    before = model.fedavg_train(
        split.clients_X, split.clients_y, cfg.fl_rounds, cfg.local_epochs, cfg.lr, cfg.batch_size, seed=seed
    )
    return model, before


def run_trial(pop: Population, cfg: TrialConfig, seed: int) -> dict:
    split = split_population(pop, cfg.k_clients, cfg.calib_frac, cfg.holdout_frac, seed=seed)
    model, before = _fit_base_model(pop, split, cfg, seed=seed)

    rng = np.random.default_rng(seed + 999)
    fc = cfg.forget_client
    n_local = len(split.clients_X[fc])
    n_forget = max(1, int(n_local * cfg.forget_frac))
    forget_idx = rng.choice(n_local, size=n_forget, replace=False)
    forget_mask = np.zeros(n_local, dtype=bool)
    forget_mask[forget_idx] = True
    forget_X = split.clients_X[fc][forget_mask]
    forget_y = split.clients_y[fc][forget_mask]

    def unlearn_fn(Xf, yf, before_params, rep_seed):
        # Calibration replicate, run server-side (never seen by the client):
        # a FRESH full retrain of the whole federation, structured exactly
        # like the real one (same base seed and per-client ordering, so
        # SGD-trajectory noise unrelated to the forget request cancels out),
        # with the calibration subset Xf standing in for the true forgotten
        # records inside the forget-client's shard -- then the same honest
        # unlearning step used for real. This mirrors how the real
        # forgotten records were genuinely baked into the model from round
        # zero; a late warm-start of Xf onto an already-converged model
        # under-integrates it and gives a degenerate, near-zero null, and a
        # fresh but differently-seeded/ordered retrain adds spurious
        # trajectory variance that swamps the actual forgetting signal.
        kept_X = split.clients_X[fc][~forget_mask]
        kept_y = split.clients_y[fc][~forget_mask]
        shard_X = np.concatenate([Xf, kept_X])
        shard_y = np.concatenate([yf, kept_y])
        synth_mask = np.zeros(len(shard_X), dtype=bool)
        synth_mask[: len(Xf)] = True
        calib_clients_X = list(split.clients_X)
        calib_clients_X[fc] = shard_X
        calib_clients_y = list(split.clients_y)
        calib_clients_y[fc] = shard_y
        fresh = FederatedSoftmax(pop.n_features, pop.n_classes, model.l2)
        fresh.init(seed=seed)
        before_ref = model.fedavg_train(
            calib_clients_X, calib_clients_y, cfg.fl_rounds, cfg.local_epochs, cfg.lr, cfg.batch_size,
            seed=seed, init_params=fresh.params,
        )
        after_ref = honest_finetune_unlearn(
            model, calib_clients_X, calib_clients_y, fc, synth_mask, before_ref, rep_seed,
            cfg.unlearn_extra_rounds, cfg.local_epochs, cfg.lr, cfg.batch_size,
        )
        return before_ref, after_ref

    after_honest = honest_finetune_unlearn(
        model, split.clients_X, split.clients_y, fc, forget_mask, before, seed,
        cfg.unlearn_extra_rounds, cfg.local_epochs, cfg.lr, cfg.batch_size,
    )
    after_lazy = dishonest_lazy(
        model, split.clients_X, split.clients_y, fc, forget_mask, before, seed,
        cfg.unlearn_extra_rounds, cfg.local_epochs, cfg.lr, cfg.batch_size,
    )
    after_spoof = dishonest_spoof(
        model, split.clients_X, split.clients_y, fc, forget_mask, before, seed, cfg.spoof_epochs, cfg.spoof_lr
    )

    retained_X = np.concatenate(
        [split.clients_X[k] if k != fc else split.clients_X[fc][~forget_mask] for k in range(cfg.k_clients)]
    )
    retained_y = np.concatenate(
        [split.clients_y[k] if k != fc else split.clients_y[fc][~forget_mask] for k in range(cfg.k_clients)]
    )
    member_X, member_y = nearest_neighbor_pool(forget_X, retained_X, retained_y, cfg.n_neighbors_leak)
    nonmember_X, nonmember_y = nearest_neighbor_pool(forget_X, split.holdout_X, split.holdout_y, cfg.n_neighbors_leak)

    out = {"seed": seed}

    # --- baseline 1: parameter distance ---
    out["pdiff_honest"] = parameter_diff_statistic(before, after_honest)
    out["pdiff_lazy"] = parameter_diff_statistic(before, after_lazy)
    out["pdiff_spoof"] = parameter_diff_statistic(before, after_spoof)

    # --- baseline 2: unrestricted behavioral audit (broad retained-set queries) ---
    out["ub_honest"] = unrestricted_behavioral_statistic(model, retained_X, retained_y, before, after_honest, cfg.n_query_unrestricted, seed)
    out["ub_lazy"] = unrestricted_behavioral_statistic(model, retained_X, retained_y, before, after_lazy, cfg.n_query_unrestricted, seed)
    out["ub_spoof"] = unrestricted_behavioral_statistic(model, retained_X, retained_y, before, after_spoof, cfg.n_query_unrestricted, seed)
    out["ub_leak_honest"] = leakage_auc(model, member_X, member_y, nonmember_X, nonmember_y, before, after_honest, sigma=0.0, l_max=cfg.l_max, seed=seed)

    # --- SEAL, swept over epsilon budgets ---
    # Calibration replicates (the expensive part -- one fresh full retrain
    # per replicate) do not depend on epsilon, so build them once and reuse
    # across the whole sweep; only the noise applied to channel B changes.
    calib_replicates = build_calibration_replicates(
        split.calib_X, split.calib_y, before, unlearn_fn, cfg.n_calib_reps, seed, m=len(forget_X)
    )
    for eps in [0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 1e6]:
        cert = SealCertificate(
            model, l_max=cfg.l_max, n_probes=cfg.n_probes, jitter=cfg.jitter,
            epsilon_audit=eps, delta_audit=cfg.delta_audit, beta_target=cfg.beta_target,
        )
        res_honest = cert.audit(forget_X, forget_y, before, after_honest, calib_replicates, seed=seed)
        res_lazy = cert.audit(forget_X, forget_y, before, after_lazy, calib_replicates, seed=seed)
        res_spoof = cert.audit(forget_X, forget_y, before, after_spoof, calib_replicates, seed=seed)
        tag = f"eps{eps:g}"
        out[f"seal_{tag}_lamB_honest"] = res_honest.channel_b.lam
        out[f"seal_{tag}_lamB_lazy"] = res_lazy.channel_b.lam
        out[f"seal_{tag}_lamB_spoof"] = res_spoof.channel_b.lam
        out[f"seal_{tag}_lamA_honest"] = res_honest.channel_a.lam
        out[f"seal_{tag}_lamA_lazy"] = res_lazy.channel_a.lam
        out[f"seal_{tag}_lamA_spoof"] = res_spoof.channel_a.lam
        out[f"seal_{tag}_flag_honest"] = res_honest.decision_dishonest
        out[f"seal_{tag}_flag_lazy"] = res_lazy.decision_dishonest
        out[f"seal_{tag}_flag_spoof"] = res_spoof.decision_dishonest
        out[f"seal_{tag}_leak"] = leakage_auc(model, member_X, member_y, nonmember_X, nonmember_y, before, after_honest, sigma=cert.budget.sigma, l_max=cfg.l_max, seed=seed)

    return out


def run_sweep(pop: Population, cfg: TrialConfig, n_trials: int, seed0: int = 0) -> pd.DataFrame:
    rows = [run_trial(pop, cfg, seed0 + t) for t in range(n_trials)]
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame, eps_list) -> pd.DataFrame:
    """Collapse the per-trial dataframe into the (leakage, detection power)
    Pareto-frontier table used for the report and plots."""
    rows = []
    # baseline: parameter diff -- detection power via Monte-Carlo AUC, leakage ~ 0 by construction
    rows.append({
        "method": "parameter-diff",
        "epsilon": float("nan"),
        "leakage_auc": 0.5,
        "power_lazy": detection_power(df["pdiff_honest"].values, df["pdiff_lazy"].values),
        "power_spoof": detection_power(df["pdiff_honest"].values, df["pdiff_spoof"].values),
    })
    # baseline: unrestricted behavioral audit -- full power reference point, leakage measured empirically
    rows.append({
        "method": "unrestricted-behavioral",
        "epsilon": float("inf"),
        "leakage_auc": float(df["ub_leak_honest"].mean()),
        "power_lazy": detection_power(df["ub_honest"].values, df["ub_lazy"].values),
        "power_spoof": detection_power(df["ub_honest"].values, df["ub_spoof"].values),
    })
    for eps in eps_list:
        tag = f"eps{eps:g}"
        rows.append({
            "method": "SEAL",
            "epsilon": eps,
            "leakage_auc": float(df[f"seal_{tag}_leak"].mean()),
            "power_lazy": detection_power(df[f"seal_{tag}_lamA_honest"].values, df[f"seal_{tag}_lamA_lazy"].values),
            "power_spoof": detection_power(df[f"seal_{tag}_lamB_honest"].values, df[f"seal_{tag}_lamB_spoof"].values),
            "flag_rate_honest": float(df[f"seal_{tag}_flag_honest"].mean()),
            "flag_rate_lazy": float(df[f"seal_{tag}_flag_lazy"].mean()),
            "flag_rate_spoof": float(df[f"seal_{tag}_flag_spoof"].mean()),
        })
    return pd.DataFrame(rows)
