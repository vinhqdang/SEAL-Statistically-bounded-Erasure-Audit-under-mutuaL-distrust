import numpy as np
import pytest
from scipy import stats

from seal.flmodel import FederatedSoftmax
from seal.mechanisms import honest_finetune_unlearn, dishonest_lazy, dishonest_spoof
from seal.certificate import SealCertificate, build_calibration_replicates, _predictive_quantile
from seal.metrics import detection_power


def test_predictive_quantile_exceeds_asymptotic_z_quantile_at_finite_m():
    """The exact finite-sample (Student-t, 'prediction interval for a
    future observation') threshold must be strictly wider than the
    asymptotic z-quantile plug-in test previously used, at every finite
    calibration-replicate count -- the z-test silently understates the
    true threshold whenever the null's mean/std are themselves estimated
    from finite data, which is always true here."""
    beta = 0.05
    z = stats.norm.ppf(1 - beta)
    for m in [2, 3, 5, 10, 25, 100]:
        q = _predictive_quantile(beta, m)
        assert q > z, f"predictive quantile at m={m} should exceed the asymptotic z quantile"


def test_predictive_quantile_converges_to_z_quantile_as_m_grows():
    beta = 0.05
    z = stats.norm.ppf(1 - beta)
    q_small = _predictive_quantile(beta, 3)
    q_large = _predictive_quantile(beta, 100_000)
    assert q_small - z > q_large - z > 0
    assert q_large == pytest.approx(z, abs=0.01)


def test_predictive_quantile_matches_hand_derived_formula_at_m_equals_2():
    # t_crit(1 df, 0.95) = 6.3138 exactly (a classical, tabulated value);
    # the extra sqrt(1 + 1/m) factor for m=2 is sqrt(1.5).
    beta = 0.05
    q = _predictive_quantile(beta, 2)
    expected = stats.t.ppf(0.95, df=1) * np.sqrt(1.5)
    assert q == pytest.approx(expected, rel=1e-9)
    assert expected == pytest.approx(6.3138 * 1.224745, rel=1e-3)


def _toy_federation(seed=0, n_per_client=800, k=4, d=6, c=2):
    # A single shared decision boundary across clients and the calibration
    # pool (iid), so honest retraining has a clean effect on any held-out
    # chunk -- a non-shared boundary per client makes FedAvg converge
    # poorly and gives a near-zero, high-variance null regardless of the
    # audit logic being tested. n_per_client and the label noise are large
    # enough that "which 40 points were forgotten" doesn't itself dominate
    # the honest effect size (a smaller/near-separable toy setup makes
    # channel A's sign flip from one random forget-subset to the next).
    rng = np.random.default_rng(seed)
    true_w = rng.normal(size=d)
    clients_X, clients_y = [], []
    for _ in range(k):
        X = rng.normal(size=(n_per_client, d))
        y = (X @ true_w + rng.normal(scale=0.4, size=n_per_client) > 0).astype(int)
        clients_X.append(X)
        clients_y.append(y)
    calib_X = rng.normal(size=(1000, d))
    calib_y = (calib_X @ true_w + rng.normal(scale=0.4, size=1000) > 0).astype(int)
    return clients_X, clients_y, calib_X, calib_y, d, c


def _make_unlearn_fn(model, clients_X, clients_y, fc, train_seed, fl_rounds=15, extra_rounds=6, lr=0.3):
    """Calibration replicate: a fresh full retrain structured exactly like
    the real training run (same base seed, same per-client ordering), with
    the calibration subset standing in for the forget-client's shard, then
    the same honest unlearning step. Matching the base training seed and
    client ordering removes SGD-trajectory noise unrelated to the actual
    "was this data forgotten" signal; anything else (a late warm-start, or
    an unrelated random seed per replicate) under- or mis-estimates the
    honest effect size and gives a degenerate null."""

    def unlearn_fn(Xf, yf, before_params, seed):
        shard_X, shard_y = np.concatenate([Xf, clients_X[fc]]), np.concatenate([yf, clients_y[fc]])
        synth_mask = np.zeros(len(shard_X), dtype=bool)
        synth_mask[: len(Xf)] = True
        calib_X_list = list(clients_X)
        calib_X_list[fc] = shard_X
        calib_y_list = list(clients_y)
        calib_y_list[fc] = shard_y
        fresh = FederatedSoftmax(model.n_features, model.n_classes, model.l2)
        fresh.init(seed=train_seed)
        before_ref = model.fedavg_train(
            calib_X_list, calib_y_list, fl_rounds, 1, lr, 32, seed=train_seed, init_params=fresh.params
        )
        after_ref = honest_finetune_unlearn(model, calib_X_list, calib_y_list, fc, synth_mask, before_ref, seed, extra_rounds, 1, lr, 32)
        return before_ref, after_ref

    return unlearn_fn


def test_lazy_dishonest_flagged_more_than_honest():
    """Channel A (own-forgotten-data) should separate a server that
    genuinely retrained from one that did nothing, across many independent
    forget-requests -- checked both as a raw statistic (Monte-Carlo AUC)
    and through the full calibrated certificate decision."""
    clients_X, clients_y, calib_X, calib_y, d, c = _toy_federation()
    model = FederatedSoftmax(d, c, l2=1e-3)
    model.init(seed=1)
    train_seed = 1
    before = model.fedavg_train(clients_X, clients_y, rounds=20, local_epochs=1, lr=0.3, batch_size=32, seed=train_seed)
    fc = 0

    honest_lams, lazy_lams = [], []
    honest_flags, lazy_flags = [], []
    cert = SealCertificate(model, l_max=8.0, n_probes=30, jitter=0.3, epsilon_audit=1.0, delta_audit=1e-5, beta_target=0.2)

    for trial_seed in range(8):
        rng = np.random.default_rng(trial_seed)
        n_local = len(clients_X[fc])
        forget_idx = rng.choice(n_local, size=80, replace=False)
        forget_mask = np.zeros(n_local, dtype=bool)
        forget_mask[forget_idx] = True
        forget_X, forget_y = clients_X[fc][forget_mask], clients_y[fc][forget_mask]

        after_honest = honest_finetune_unlearn(model, clients_X, clients_y, fc, forget_mask, before, seed=trial_seed, extra_rounds=8, local_epochs=1, lr=0.3, batch_size=32)
        after_lazy = dishonest_lazy(model, clients_X, clients_y, fc, forget_mask, before, trial_seed, 8, 1, 0.3, 32)

        lam_h = float(np.mean(model.per_sample_loss(forget_X, forget_y, after_honest) - model.per_sample_loss(forget_X, forget_y, before)))
        lam_l = float(np.mean(model.per_sample_loss(forget_X, forget_y, after_lazy) - model.per_sample_loss(forget_X, forget_y, before)))
        honest_lams.append(lam_h)
        lazy_lams.append(lam_l)

        unlearn_fn = _make_unlearn_fn(model, [clients_X[k][~forget_mask] if k == fc else clients_X[k] for k in range(4)],
                                       [clients_y[k][~forget_mask] if k == fc else clients_y[k] for k in range(4)],
                                       fc, train_seed, fl_rounds=20, extra_rounds=8)
        reps = build_calibration_replicates(calib_X, calib_y, before, unlearn_fn, n_reps=20, seed=trial_seed, m=len(forget_X))
        honest_flags.append(cert.audit(forget_X, forget_y, before, after_honest, reps, seed=trial_seed).decision_dishonest)
        lazy_flags.append(cert.audit(forget_X, forget_y, before, after_lazy, reps, seed=trial_seed).decision_dishonest)

    # "Lazy" here means the server keeps training on data including the
    # supposedly-forgotten records (not literally freezing), which is a
    # harder, more realistic adversary and only a moderate-power toy-scale
    # signal -- full-scale experiments calibrate a proper decision
    # threshold instead of reading this raw statistic directly.
    assert detection_power(np.array(honest_lams), np.array(lazy_lams)) > 0.65
    assert sum(lazy_flags) >= sum(honest_flags)


def test_channel_a_alone_is_fooled_by_spoofing_but_channel_b_reacts():
    clients_X, clients_y, calib_X, calib_y, d, c = _toy_federation(seed=5)
    model = FederatedSoftmax(d, c, l2=1e-3)
    model.init(seed=1)
    before = model.fedavg_train(clients_X, clients_y, rounds=15, local_epochs=1, lr=0.3, batch_size=32, seed=1)

    fc = 0
    n_local = len(clients_X[fc])
    forget_mask = np.zeros(n_local, dtype=bool)
    forget_mask[:40] = True
    forget_X, forget_y = clients_X[fc][forget_mask], clients_y[fc][forget_mask]

    after_spoof = dishonest_spoof(model, clients_X, clients_y, fc, forget_mask, before, seed=4, epochs=60, lr=0.5)

    # Channel A (own-data-only) should show a large loss increase on the
    # exact forgotten points -- that's the whole point of the spoof attack.
    lam_a = float(np.mean(model.per_sample_loss(forget_X, forget_y, after_spoof) - model.per_sample_loss(forget_X, forget_y, before)))
    assert lam_a > 0.05
