"""End-to-end SEAL-W experiment, TOFU-grounded variant of
`run_llm_experiment.py`: same certificate, same audited mechanisms, same
watermarking machinery (`seal_llm.watermark.generate_watermarked`,
completely unchanged) -- the only difference is where the author pool comes
from. `run_llm_experiment.py` invents ten fictitious names and templated
biography prompts out of thin air (`seal_llm.data.AUTHOR_NAMES`,
`build_authors`); this script instead draws real fictitious-author names and
short real biographical fact snippets from the actual, published TOFU
unlearning benchmark (`locuslab/TOFU` on HuggingFace) via
`seal_llm.tofu_data.load_tofu_authors` and `seal_llm.data.build_tofu_authors`.

IMPORTANT -- what this substitution is, and is not:
  * It is NOT training on TOFU's own question/answer text. TOFU's answers
    are static, pre-existing GPT-4-generated text with no watermark bias
    baked into them; using them as training documents directly would carry
    no green-list signal at all and would silently break the certificate's
    entire detection mechanism.
  * It IS: real TOFU author names, and a short real biographical fact
    snippet extracted from that author's real TOFU QA rows, spliced into a
    seed prompt (`f"{name}, {fact_snippet}, is an author who"`) that is
    handed to the exact same, unmodified `generate_watermarked` function
    `run_llm_experiment.py` uses. Every training document produced here is
    still our own watermarked generation -- the certificate mechanism is
    identical -- just grounded in a real entity and a real fact about them
    instead of an invented one. See `seal_llm.data.build_tofu_authors` and
    `seal_llm.tofu_data.load_tofu_authors` for the exact heuristics.

IMPORTANT -- comparability to the official TOFU benchmark:
  This project only ever fine-tunes `distilgpt2` (CPU-only, no GPU). The
  official TOFU leaderboard fine-tunes and evaluates phi-1.5 (~1.3B params)
  or llama2-7b using TOFU's own forget-quality/model-utility metrics. NONE
  of the numbers this script produces are comparable to published TOFU
  leaderboard numbers -- different model, different (much smaller) scale,
  and an entirely different detection mechanism (a watermark z-score/PIR
  certificate, not TOFU's own utility/forget-quality metrics). The only
  thing borrowed from TOFU here is real fictitious author names and real
  biographical facts about them, used purely as generation seeds.

Negative-canary "phantom" authors: `run_llm_experiment.py` hand-invents four
phantom names (`PHANTOM_NAMES`) that never appear in training, purely as a
negative control. Here, phantom authors are instead drawn from the SAME
real TOFU distribution as the trainable pool (so they are realistic
fictitious names from the actual benchmark, not hand-invented ones) but are
reserved into their own pool up front and never appear in ANY trial's
training data -- not the forgotten author, not any retained author, not any
calibration author -- across the whole run (see `_load_tofu_pools` below).

Usage: python3 scripts/run_llm_experiment_tofu.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from seal_llm.data import PARAPHRASE_TEMPLATES, build_tofu_authors
from seal_llm.tofu_data import load_tofu_authors
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
# Same validated config as run_llm_experiment.py (distilgpt2, ~48
# watermarked training instances/author, 3 epochs) -- see that script's own
# comment and docs/RESULTS_LLM.md for the reference run this was checked
# against. Kept identical here since the watermarking/fine-tuning machinery
# is completely unchanged; only the author *source* differs in this script.
N_AUTHORS_PER_TRIAL = 5     # subsampled from the trainable TOFU pool each trial: 1 forget + N_CALIB_AUTHORS calib + >=2 retained
N_DOCS = 4
DOC_REPEATS = 8
FT_EPOCHS = 3
FT_LR = 5e-5
GEN_MAX_NEW_TOKENS = 30
BETA_TARGET = 0.10
N_CALIB_AUTHORS = 2  # needs >=2 for a defined sample std (ddof=1) in the null
N_DECOYS = 2         # reduced from the validated-signal reference config to keep wall-clock time reasonable on this CPU-only box
REAL_CLUSTER_SIZE = len(PARAPHRASE_TEMPLATES)
PHANTOM_CLUSTER_SIZE = len(PARAPHRASE_TEMPLATES)
PARTIAL_GUESS_FRACTION = 0.5

# TOFU author pool sizes -- mirrors len(AUTHOR_NAMES)==10 and
# len(PHANTOM_NAMES)==4 in the synthetic script's fixed pools. Fetched ONCE
# (see _load_tofu_pools) so the phantom pool is reserved before any trial
# runs and stays disjoint from every trial's training data for the whole run.
TOFU_POOL_SIZE = 10
TOFU_PHANTOM_POOL_SIZE = 4
TOFU_SAMPLE_SEED = 0  # fixed: the pool split itself is chosen once, not per-trial

# Reduced from run_llm_experiment.py's N_TRIALS=3 to 2, purely for CPU
# wall-clock reasons on this box -- each trial fine-tunes distilgpt2 from
# scratch twice (retain-all + honest-unlearn) plus once per calibration
# author, and this script additionally pays a one-time TOFU dataset
# download/load. Same documented tradeoff as N_DECOYS above, not a claim
# that 2 trials is statistically sufficient on its own.
N_TRIALS = 2


def _load_tofu_pools():
    """Fetch `TOFU_POOL_SIZE + TOFU_PHANTOM_POOL_SIZE` distinct real TOFU
    authors ONCE (seeded, so this split is deterministic and reproducible),
    then split them into a trainable pool (subsampled per trial, exactly
    like AUTHOR_NAMES) and a phantom pool (reserved: used only to build
    negative-canary queries, never fine-tuned on, for every trial in this
    run)."""
    all_tofu_authors = load_tofu_authors(n_authors=TOFU_POOL_SIZE + TOFU_PHANTOM_POOL_SIZE, seed=TOFU_SAMPLE_SEED)
    trainable_pool = all_tofu_authors[:TOFU_POOL_SIZE]
    phantom_pool = all_tofu_authors[TOFU_POOL_SIZE:]
    return trainable_pool, phantom_pool


def _canonical_watermark_z(resp, prompt: str, key: int, tokenizer, gen_seed: int) -> float:
    """The WaterDrum-equivalent check: score the response to the ONE
    literal canonical (document-seed) prompt, no paraphrasing -- see
    run_llm_experiment.py's identical helper."""
    text = resp(prompt, gen_seed)
    prev_token_id = tokenizer(prompt, return_tensors="pt").input_ids[0, -1].item()
    return score_text(tokenizer, text, prev_token_id, key=key, gamma=GAMMA).z


def run_trial(base_model, tokenizer, seed: int, trainable_pool: list, phantom_pool: list) -> dict:
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(trainable_pool))
    tofu_authors = [trainable_pool[i] for i in perm[:N_AUTHORS_PER_TRIAL]]
    names = [ta.name for ta in tofu_authors]

    forget_name = names[0]
    calib_names = names[1:1 + N_CALIB_AUTHORS]

    authors = build_tofu_authors(base_model, tokenizer, tofu_authors, gamma=GAMMA, delta=DELTA,
                                  n_docs=N_DOCS, max_new_tokens=GEN_MAX_NEW_TOKENS, seed0=seed)
    # Reserved TOFU authors, never fine-tuned on in ANY trial -- pure
    # negative-control canaries, same role PHANTOM_NAMES plays in the
    # synthetic script, but drawn from the real TOFU distribution instead
    # of hand-invented.
    phantom_authors = build_tofu_authors(base_model, tokenizer, phantom_pool, gamma=GAMMA, delta=DELTA,
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
    trainable_pool, phantom_pool = _load_tofu_pools()
    print(f"TOFU trainable pool: {[a.name for a in trainable_pool]}")
    print(f"TOFU reserved phantom pool (never trained on): {[a.name for a in phantom_pool]}")

    rows = []
    for t in range(N_TRIALS):
        t0 = time.time()
        row = run_trial(base_model, tokenizer, seed=t, trainable_pool=trainable_pool, phantom_pool=phantom_pool)
        print(f"trial {t} done in {time.time()-t0:.1f}s: "
              f"honest={row['honest_flag']} lazy={row['lazy_flag']} filter={row['filter_flag']} "
              f"topic={row['topic_flag']} partial={row['partial_flag']} naive={row['naive_filter_flag']}")
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "llm_watermark_raw_tofu.csv", index=False)

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
        "waterdrum_false_accusation_rate_honest": df["honest_waterdrum_flag"].mean(),
        "waterdrum_detection_rate_lazy": df["lazy_waterdrum_flag"].mean(),
        "waterdrum_detection_rate_filter_only": df["filter_waterdrum_flag"].mean(),
        "waterdrum_detection_rate_topic_filter": df["topic_waterdrum_flag"].mean(),
        "waterdrum_detection_rate_partial_topic_filter": df["partial_waterdrum_flag"].mean(),
    }
    print(summary)
    pd.Series(summary).to_csv(RESULTS_DIR / "llm_watermark_summary_tofu.csv")


if __name__ == "__main__":
    main()
