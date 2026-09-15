"""End-to-end SEAL-W experiment: watermark-based erasure certificate for a
small fine-tuned LLM, audited via real Paillier PIR (real-cluster + a
response-diversity channel) against honest and three dishonest servers of
increasing sophistication.

Usage: python3 scripts/run_llm_experiment.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from seal_llm.data import AUTHOR_NAMES, PHANTOM_NAMES, PARAPHRASE_TEMPLATES, build_authors
from seal_llm.model import load_base
from seal_llm.mechanisms import (
    honest_unlearn, train_retain_all, dishonest_lazy, dishonest_filter_only,
    dishonest_topic_filter, dishonest_partial_topic_filter,
)
from seal_llm.certificate import (
    build_candidate_batch, commit_batch, evaluate_batch, decide, evasion_probability,
    waterdrum_baseline_decide,
)
from seal_llm.watermark import score_text

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

GAMMA = 0.5
DELTA = 8.0
# Validated against a manual reference run (distilgpt2, ~48 watermarked
# training instances/author, 3 epochs: mean z~5.25 fine-tuned vs ~0.03
# unseen) and against this repo's own smoke tests: doc_repeats=8, epochs=3
# cleanly separates honest from lazy and gives the positive canary a
# reliably strong signature. Smaller repeats/epochs leave the watermark
# too weak for the canary sanity check to pass reliably, which degenerates
# the certificate into flagging everything regardless of the true
# mechanism -- see docs/RESULTS_LLM.md for that failure mode measured
# directly, rather than only asserted here.
N_AUTHORS_PER_TRIAL = 9     # subsampled from AUTHOR_NAMES each trial: 1 forget + N_CALIB_AUTHORS calib + >=2 retained
N_DOCS = 4
DOC_REPEATS = 8
FT_EPOCHS = 3
FT_LR = 5e-5
GEN_MAX_NEW_TOKENS = 30
BETA_TARGET = 0.10
# Bumped from N_CALIB_AUTHORS=2 (the bare minimum for a defined sample std)
# once GPU acceleration made the extra fine-tune replicates affordable: this
# paper's own central diagnosis (Section 6.1.1) and the classical-track
# ablation (Table tab:calibablation) both show N_calib=2 produces an
# unstable, occasionally wildly-oversized threshold (Theorem 3's exact
# Student-t correction has only 1 degree of freedom there) that suppresses
# genuine detection power -- not a claim that the certificate mechanism
# itself is broken. N_CALIB_AUTHORS=6 (5 degrees of freedom) is the
# principled fix the paper already argues for, applied here rather than
# only argued for elsewhere; it requires N_AUTHORS_PER_TRIAL=9 to leave
# room for >=2 retained authors from the 10-name AUTHOR_NAMES pool.
N_CALIB_AUTHORS = 6
N_DECOYS = 2         # reduced from the validated-signal reference config to keep wall-clock time reasonable
# Both clusters use the FULL paraphrase-template pool (rather than a random
# subset of it) so the analytic hypergeometric prediction below (N = pool
# size, exactly one slot checked, m = guessed templates) matches what
# dishonest_partial_topic_filter actually does one-for-one.
REAL_CLUSTER_SIZE = len(PARAPHRASE_TEMPLATES)
PHANTOM_CLUSTER_SIZE = len(PARAPHRASE_TEMPLATES)
PARTIAL_GUESS_FRACTION = 0.5
# Bumped from the original N_TRIALS=3 specifically because reviewers correctly
# flagged that a rate like 1/3 carries a 95% Clopper-Pearson interval of
# roughly [0.01, 0.91] -- not enough trials to distinguish signal from noise.
# Tables tab:llmfull and tab:waterdrum both reflect this full N_TRIALS=30,
# N_CALIB_AUTHORS=6 configuration (results/llm_watermark_raw.csv / _summary.csv),
# assembled from four separate GPU runs across three Colab accounts after
# repeated mid-run disconnects (an infrastructure inconvenience, not a data
# quality issue: every row is a complete, real trial, recovered via
# incremental per-trial logging rather than lost to the disconnects).
# Re-running this script end-to-end reproduces the same configuration
# in one pass.
N_TRIALS = 30


def _canonical_watermark_z(resp, prompt: str, key: int, tokenizer, gen_seed: int) -> float:
    """The WaterDrum-equivalent check: score the response to the ONE
    literal canonical (document-seed) prompt, no paraphrasing."""
    text = resp(prompt, gen_seed)
    prev_token_id = tokenizer(prompt, return_tensors="pt").input_ids[0, -1].item()
    return score_text(tokenizer, text, prev_token_id, key=key, gamma=GAMMA).z


def run_trial(base_model, tokenizer, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(AUTHOR_NAMES))
    names = [AUTHOR_NAMES[i] for i in perm[:N_AUTHORS_PER_TRIAL]]

    forget_name = names[0]
    calib_names = names[1:1 + N_CALIB_AUTHORS]

    authors = build_authors(base_model, tokenizer, names, gamma=GAMMA, delta=DELTA,
                             n_docs=N_DOCS, max_new_tokens=GEN_MAX_NEW_TOKENS, seed0=seed)
    phantom_authors = build_authors(base_model, tokenizer, PHANTOM_NAMES, gamma=GAMMA, delta=DELTA,
                                     n_docs=1, max_new_tokens=GEN_MAX_NEW_TOKENS, seed0=seed + 5000)

    train_kwargs = dict(epochs=FT_EPOCHS, lr=FT_LR, doc_repeats=DOC_REPEATS)
    mech_kwargs = dict(**train_kwargs, gen_max_new_tokens=GEN_MAX_NEW_TOKENS)

    # One shared fine-tune stands in for every "kept training on
    # everything, never actually unlearned" dishonest server -- lazy,
    # filter_only, topic_filter and partial_topic_filter differ only in
    # their output-filtering wrapper, not in what the weights learned.
    retain_all_model = train_retain_all(base_model, tokenizer, authors, seed=seed, **train_kwargs)
    _, resp_honest = honest_unlearn(base_model, tokenizer, authors, forget_name, seed=seed, **mech_kwargs)
    resp_lazy = dishonest_lazy(retain_all_model, tokenizer, gen_max_new_tokens=GEN_MAX_NEW_TOKENS)
    forget_prompt = next(a.prompt for a in authors if a.name == forget_name)
    resp_filter = dishonest_filter_only(retain_all_model, tokenizer, forget_prompt, gen_max_new_tokens=GEN_MAX_NEW_TOKENS)
    resp_topic = dishonest_topic_filter(retain_all_model, tokenizer, forget_name, gen_max_new_tokens=GEN_MAX_NEW_TOKENS)
    resp_partial = dishonest_partial_topic_filter(retain_all_model, tokenizer, forget_name,
                                                   guess_fraction=PARTIAL_GUESS_FRACTION, rng_seed=seed,
                                                   gen_max_new_tokens=GEN_MAX_NEW_TOKENS)

    out = {"seed": seed, "forget_name": forget_name}

    # Null distributions (both channels), shared across all mechanism
    # comparisons for this trial: rerun the HONEST mechanism targeting each
    # calibration author instead of the real forget request.
    null_zs, null_divs, null_pos, null_neg, null_wd = [], [], [], [], []
    for calib_name in calib_names:
        _, resp_calib = honest_unlearn(base_model, tokenizer, authors, calib_name, seed=seed, **mech_kwargs)
        calib_batch = build_candidate_batch(authors, calib_name, phantom_authors, n_decoys=N_DECOYS,
                                             real_cluster_size=REAL_CLUSTER_SIZE, phantom_cluster_size=PHANTOM_CLUSTER_SIZE,
                                             seed=seed + hash(calib_name) % 997)
        calib_committed = commit_batch(resp_calib, calib_batch, gamma=GAMMA, gen_seed_base=seed * 733, tokenizer=tokenizer)
        raw = evaluate_batch(calib_committed, rng_seed=seed + hash(calib_name) % 991)
        null_zs.append(raw.z_real)
        null_divs.append(raw.diversity_ratio)
        null_pos.append(raw.z_positive_canary)
        null_neg.append(raw.z_negative_canary_mean)

        # Same calibration replicate, but scored the WaterDrum-equivalent
        # way: the calibration author's own literal canonical prompt, no
        # paraphrasing -- an independent null for the baseline comparison
        # below, built from the exact same honest re-runs.
        calib_prompt = next(a.prompt for a in authors if a.name == calib_name)
        calib_key = next(a.key for a in authors if a.name == calib_name)
        null_wd.append(_canonical_watermark_z(resp_calib, calib_prompt, calib_key, tokenizer, seed * 733))
    null_zs = np.array(null_zs)
    null_divs = np.array(null_divs)
    null_pos = np.array(null_pos)
    null_neg = np.array(null_neg)
    null_wd = np.array(null_wd)

    forget_key = next(a.key for a in authors if a.name == forget_name)

    mechanisms = [
        ("honest", resp_honest), ("lazy", resp_lazy), ("filter", resp_filter),
        ("topic", resp_topic), ("partial", resp_partial),
    ]
    for tag, resp in mechanisms:
        batch = build_candidate_batch(authors, forget_name, phantom_authors, n_decoys=N_DECOYS,
                                       real_cluster_size=REAL_CLUSTER_SIZE, phantom_cluster_size=PHANTOM_CLUSTER_SIZE,
                                       seed=seed)
        committed = commit_batch(resp, batch, gamma=GAMMA, gen_seed_base=seed * 131 + hash(tag) % 1000, tokenizer=tokenizer)
        raw = evaluate_batch(committed, rng_seed=seed * 17 + hash(tag) % 983)
        res = decide(raw, null_zs, null_divs, beta_target=BETA_TARGET,
                     null_positive_canary=null_pos, null_negative_canary=null_neg)
        out[f"{tag}_z_real"] = res.z_real
        out[f"{tag}_z_pos"] = res.z_positive_canary
        out[f"{tag}_z_neg"] = res.z_negative_canary_mean
        out[f"{tag}_threshold"] = res.threshold
        out[f"{tag}_diversity_ratio"] = res.diversity_ratio
        out[f"{tag}_diversity_threshold"] = res.diversity_threshold
        out[f"{tag}_diversity_flag"] = res.diversity_flag
        out[f"{tag}_flag"] = res.decision_dishonest

        # WaterDrum-equivalent baseline on the SAME mechanism's response:
        # one literal canonical prompt, slot-channel-only, no diversity
        # channel, no real-cluster -- see
        # seal_llm.certificate.WaterDrumBaselineResult's docstring.
        wd_z = _canonical_watermark_z(resp, forget_prompt, forget_key, tokenizer, seed * 131 + hash(tag) % 1000)
        wd_res = waterdrum_baseline_decide(wd_z, null_wd, beta_target=BETA_TARGET)
        out[f"{tag}_waterdrum_z"] = wd_res.z
        out[f"{tag}_waterdrum_threshold"] = wd_res.threshold
        out[f"{tag}_waterdrum_flag"] = wd_res.flag

    # Naive baseline (no PIR, no clustering, no paraphrasing): the auditor
    # asks the filtering server the ONE known canonical trigger prompt
    # directly. Demonstrates the failure mode the real protocol avoids --
    # the exact-match filter catches it and the naive check reads null.
    prev_token_id = tokenizer(forget_prompt, return_tensors="pt").input_ids[0, -1].item()
    naive_text = resp_filter(forget_prompt, seed * 131)
    naive_score = score_text(tokenizer, naive_text, prev_token_id, key=forget_key, gamma=GAMMA)
    out["naive_filter_z_real"] = naive_score.z
    out["naive_filter_flag"] = bool(naive_score.z > out["filter_threshold"])

    return out


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    base_model, tokenizer = load_base("distilgpt2")
    rows = []
    for t in range(N_TRIALS):
        t0 = time.time()
        row = run_trial(base_model, tokenizer, seed=t)
        print(f"trial {t} done in {time.time()-t0:.1f}s: "
              f"honest={row['honest_flag']} lazy={row['lazy_flag']} filter={row['filter_flag']} "
              f"topic={row['topic_flag']} partial={row['partial_flag']} naive={row['naive_filter_flag']}")
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "llm_watermark_raw.csv", index=False)

    summary = {
        "false_accusation_rate_honest": df["honest_flag"].mean(),
        "detection_rate_lazy": df["lazy_flag"].mean(),
        "detection_rate_filter_only": df["filter_flag"].mean(),
        "detection_rate_topic_filter": df["topic_flag"].mean(),
        "detection_rate_topic_filter_diversity_channel_alone": df["topic_diversity_flag"].mean(),
        "detection_rate_partial_topic_filter": df["partial_flag"].mean(),
        "naive_no_pir_detection_rate_filter_only": df["naive_filter_flag"].mean(),
        "analytic_evasion_prob_partial_filter_within_cluster":
            evasion_probability(n_slots=len(PARAPHRASE_TEMPLATES), n_protected=1,
                                 n_tampered=max(1, round(PARTIAL_GUESS_FRACTION * len(PARAPHRASE_TEMPLATES)))),
        # WaterDrum-equivalent baseline (single canonical prompt, slot
        # channel only) vs full SEAL-W, mechanism by mechanism -- the
        # explicit "reproduce and break WaterDrum" comparison.
        "waterdrum_false_accusation_rate_honest": df["honest_waterdrum_flag"].mean(),
        "waterdrum_detection_rate_lazy": df["lazy_waterdrum_flag"].mean(),
        "waterdrum_detection_rate_filter_only": df["filter_waterdrum_flag"].mean(),
        "waterdrum_detection_rate_topic_filter": df["topic_waterdrum_flag"].mean(),
        "waterdrum_detection_rate_partial_topic_filter": df["partial_waterdrum_flag"].mean(),
    }
    print(summary)
    pd.Series(summary).to_csv(RESULTS_DIR / "llm_watermark_summary.csv")


if __name__ == "__main__":
    main()
