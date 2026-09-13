"""Run the SEAL vs. SOTA-baseline comparison on both benchmark datasets and
save the raw per-trial results, summary Pareto tables, and plots under
results/.

Usage: python3 scripts/run_experiments.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from seal.data import load_german_credit, load_mnist_subset, load_covertype
from seal.evaluate import TrialConfig, run_sweep, summarize

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
EPS_LIST = [0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 1e6]


def plot_pareto(summary: pd.DataFrame, title: str, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, power_col, subtitle in [
        (axes[0], "power_lazy", "detecting a server that ignores the request"),
        (axes[1], "power_spoof", "detecting targeted spoofing"),
    ]:
        seal_rows = summary[summary.method == "SEAL"].sort_values("leakage_auc")
        ax.plot(seal_rows.leakage_auc, seal_rows[power_col], "o-", color="#2a6f97", label="SEAL (epsilon sweep)")
        for _, row in seal_rows.iterrows():
            ax.annotate(f"eps={row.epsilon:g}", (row.leakage_auc, row[power_col]), fontsize=7, alpha=0.7,
                        xytext=(3, 3), textcoords="offset points")
        pd_row = summary[summary.method == "parameter-diff"].iloc[0]
        ax.scatter([pd_row.leakage_auc], [pd_row[power_col]], marker="s", color="#a83232", s=70, label="parameter-diff", zorder=5)
        ub_row = summary[summary.method == "unrestricted-behavioral"].iloc[0]
        ax.scatter([ub_row.leakage_auc], [ub_row[power_col]], marker="^", color="#c98a2e", s=70, label="unrestricted behavioral (Tang et al.-style)", zorder=5)
        ax.axhline(0.5, color="gray", linestyle=":", linewidth=1)
        ax.set_xlabel("retained-set membership leakage (attack AUC)")
        ax.set_ylabel("detection power (Monte-Carlo AUC)")
        ax.set_title(subtitle, fontsize=10)
        ax.set_ylim(0.4, 1.02)
        ax.set_xlim(0.45, 1.0)
        ax.legend(fontsize=7, loc="lower right")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def run_dataset(name: str, pop, cfg: TrialConfig, n_trials: int, seed0: int = 0) -> None:
    print(f"=== {name} ===  n={len(pop.X)}  features={pop.n_features}  classes={pop.n_classes}")
    t0 = time.time()
    df = run_sweep(pop, cfg, n_trials=n_trials, seed0=seed0)
    print(f"  ran {n_trials} trials in {time.time() - t0:.1f}s")
    df.to_csv(RESULTS_DIR / f"{name}_raw.csv", index=False)
    summary = summarize(df, EPS_LIST)
    summary.to_csv(RESULTS_DIR / f"{name}_summary.csv", index=False)
    print(summary.to_string(index=False))
    plot_pareto(summary, f"SEAL vs. SOTA baselines -- {name}", RESULTS_DIR / f"{name}_pareto.png")


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)

    credit_cfg = TrialConfig(
        k_clients=5, forget_client=0, forget_frac=0.2, calib_frac=0.15, holdout_frac=0.15,
        fl_rounds=25, local_epochs=1, lr=0.3, batch_size=32, unlearn_extra_rounds=8,
        spoof_epochs=40, spoof_lr=0.5, n_calib_reps=25, n_query_unrestricted=60,
        n_probes=50, jitter=0.4, l_max=8.0, beta_target=0.05, delta_audit=1e-5, n_neighbors_leak=3,
    )
    run_dataset("german_credit", load_german_credit(), credit_cfg, n_trials=20)

    mnist_cfg = TrialConfig(
        k_clients=5, forget_client=0, forget_frac=0.1, calib_frac=0.15, holdout_frac=0.15,
        fl_rounds=20, local_epochs=1, lr=0.3, batch_size=64, unlearn_extra_rounds=6,
        spoof_epochs=30, spoof_lr=0.5, n_calib_reps=15, n_query_unrestricted=60,
        n_probes=50, jitter=0.05, l_max=8.0, beta_target=0.05, delta_audit=1e-5, n_neighbors_leak=3,
    )
    run_dataset("mnist", load_mnist_subset(n_samples=5000), mnist_cfg, n_trials=12)

    # Two orders of magnitude bigger than German Credit's 1,000 rows: real
    # per-client shards of tens of thousands of rows each, not a few
    # hundred. Larger batch_size keeps the number of SGD steps -- the
    # actual driver of wall-clock time, not raw row count, since numpy
    # vectorizes each mini-batch -- comparable to the smaller datasets.
    covertype_cfg = TrialConfig(
        k_clients=5, forget_client=0, forget_frac=0.1, calib_frac=0.15, holdout_frac=0.15,
        fl_rounds=20, local_epochs=1, lr=0.3, batch_size=512, unlearn_extra_rounds=6,
        spoof_epochs=30, spoof_lr=0.5, n_calib_reps=12, n_query_unrestricted=60,
        n_probes=50, jitter=0.3, l_max=8.0, beta_target=0.05, delta_audit=1e-5, n_neighbors_leak=3,
    )
    run_dataset("covertype", load_covertype(n_samples=150_000), covertype_cfg, n_trials=10)


if __name__ == "__main__":
    main()
