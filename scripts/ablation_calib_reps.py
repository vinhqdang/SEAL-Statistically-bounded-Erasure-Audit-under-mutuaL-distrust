"""Ablation: how does the certificate's false-accusation rate (and, where
meaningful, detection power) change as the number of calibration replicates
n_calib_reps grows, holding every other German Credit hyperparameter fixed
at scripts/run_experiments.py's values? Directly tests the paper's own
repeated claim that "more calibration replicates, not a different formula,
is the fix" for the small-sample instability documented in the LLM track
and discussed in the Discussion section -- this is the cheap classical-track
sweep that claim implies but the original experiments never ran.

Usage: python3 scripts/ablation_calib_reps.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from seal.data import load_german_credit
from seal.evaluate import TrialConfig, run_sweep, summarize

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

N_CALIB_REPS_SWEEP = [3, 5, 10, 15, 20, 25]
N_TRIALS = 80  # higher than the 15-20 used elsewhere in this project specifically to
# keep outer Monte-Carlo noise (resolution 1/N_TRIALS per flag-rate estimate) well
# below the effect size this ablation is trying to resolve -- at N_TRIALS=15 an
# initial run of this sweep showed flag rates bouncing non-monotonically in ways
# indistinguishable from outer-trial noise, not a real n_calib_reps trend.
EPS_LIST = [1.0]  # a single representative mid-budget epsilon; false-accusation rate does not depend on epsilon at all


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    pop = load_german_credit()
    rows = []
    for n_calib_reps in N_CALIB_REPS_SWEEP:
        t0 = time.time()
        cfg = TrialConfig(
            k_clients=5, forget_client=0, forget_frac=0.2, calib_frac=0.15, holdout_frac=0.15,
            fl_rounds=25, local_epochs=1, lr=0.3, batch_size=32, unlearn_extra_rounds=8,
            spoof_epochs=40, spoof_lr=0.5, n_calib_reps=n_calib_reps, n_query_unrestricted=60,
            n_probes=50, jitter=0.4, l_max=8.0, beta_target=0.05, delta_audit=1e-5, n_neighbors_leak=3,
        )
        df = run_sweep(pop, cfg, n_trials=N_TRIALS, seed0=0)
        summary = summarize(df, EPS_LIST)
        seal_row = summary[(summary["method"] == "SEAL") & (summary["epsilon"] == 1.0)].iloc[0]

        tag = "eps1"
        flag_rate_honest = float(df[f"seal_{tag}_flag_honest"].mean())
        flag_rate_lazy = float(df[f"seal_{tag}_flag_lazy"].mean())
        flag_rate_spoof = float(df[f"seal_{tag}_flag_spoof"].mean())

        row = {
            "n_calib_reps": n_calib_reps,
            "flag_rate_honest": flag_rate_honest,
            "flag_rate_lazy": flag_rate_lazy,
            "flag_rate_spoof": flag_rate_spoof,
            "power_lazy_raw": seal_row["power_lazy"],
            "power_spoof_raw": seal_row["power_spoof"],
        }
        rows.append(row)
        print(f"n_calib_reps={n_calib_reps} done in {time.time()-t0:.1f}s: {row}")

    out = pd.DataFrame(rows)
    out.to_csv(RESULTS_DIR / "ablation_calib_reps.csv", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
