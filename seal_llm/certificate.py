"""The LLM-track SEAL-W certificate: watermark-based detection, retrieved
obliviously from a committed candidate batch via real Paillier PIR, with
canaries, a real-cluster of independently-phrased real-check queries, a
response-diversity channel, and a calibrated decision threshold.

Protocol, run by the auditor (data owner / regulator), against a server
that has already produced a model `M'` and claims the forgotten author's
influence is gone:

  1. COMMIT. The auditor assembles a candidate batch: a REAL-CLUSTER of
     several independently-worded paraphrase queries about the forgotten
     author (not one query -- see "Why a cluster, not one query" below), a
     positive-control canary (another author's query, known still-
     retained -- should show a strong watermark signature regardless of
     what happened to the forgotten one), a NEGATIVE-CANARY CLUSTER of
     several paraphrased queries about a phantom name never used in any
     training at all (should always read near-null, and its response
     diversity is the baseline the real-cluster's diversity is judged
     against), and further single decoy queries from other retained
     authors, padding the batch. Every query, real-cluster included, is a
     randomly chosen paraphrase (see PARAPHRASE_TEMPLATES), never the
     literal canonical document-seed string a filtering server might have
     hardcoded. The server (whichever mechanism produced M') answers EVERY
     prompt in the batch -- it cannot yet tell which few will actually be
     checked, nor which specific real-cluster member (if any) the z-score
     decision will end up reading. Each answer's watermark z-score AND a
     SimHash-style response fingerprint (for the diversity channel) are
     quantized to integers and hashed into a single sha256 commitment, so
     the server cannot later revise an answer once it learns what mattered.
  2. RETRIEVE. The auditor retrieves, via real Paillier-based PIR
     (seal_llm.pir): one RANDOMLY chosen real-cluster index (unpredictable
     to the server -- chosen only at audit time, not batch-build time),
     the positive-canary index, and every negative-canary-cluster index
     (needed for the diversity baseline). Every retrieval is
     cryptographically oblivious: the ciphertexts sent to the server are
     computationally indistinguishable from a query for any other index.
  3. DECIDE, on two independent channels:
       (a) SLOT channel: the one retrieved real-cluster z-score against a
           null built by re-running the honest mechanism on calibration
           authors (mirroring seal.certificate's calibration philosophy).
       (b) DIVERSITY channel: pairwise-Hamming diversity of the
           real-cluster's response fingerprints vs. the negative-canary
           cluster's, as a ratio, again compared against a calibrated
           null. Canaries are additional sanity/tamper checks on top of
           both.

Why a cluster, not one query for the real check
-------------------------------------------------
A server that recognizes ONE exact known trigger string is defeated by any
single paraphrase (dishonest_filter_only). A smarter server can instead
recognize the TOPIC (the forgotten author's name appears in the query,
however worded) and either (i) suppress every phrasing uniformly, or (ii)
suppress only a guessed subset of phrasings it can afford to hardcode.
Neither the wording trick nor PIR by itself defeats either of these:

* Against (i) uniform suppression, hiding WHICH slot is retrieved buys
  nothing -- the server treats every phrasing alike, so there is no
  "which one is checked" secret worth hiding. This is what the DIVERSITY
  channel is for: uniform canned replies collapse to near-identical
  fingerprints (near-zero pairwise Hamming distance), unlike genuine
  generation about an unknown topic (see dishonest_topic_filter).
* Against (ii) a resource-bounded server that guesses a subset of `m`
  phrasings out of the cluster's `N`, this IS a slot-hiding game: the
  server must guess without knowing which cluster member the auditor's
  one random retrieval lands on. See docs/ALGORITHM_LLM.md for the
  hypergeometric bound this reduces to, and `evasion_probability` below
  for the general formula (dishonest_partial_topic_filter instantiates
  this adversary concretely).

Neither channel is claimed to defeat a hypothetical adversary with a
perfect, free semantic classifier willing to also refuse ordinary public
queries about the same entity -- see docs/ALGORITHM_LLM.md for the scoped,
honest statement of what is and is not covered.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from math import comb

import numpy as np
from scipy import stats

from .data import PARAPHRASE_TEMPLATES
from .watermark import score_text
from .pir import PIRKeypair, keygen, pir_retrieve

Z_SCALE = 100  # quantization scale for encoding a float z-score as an integer
FINGERPRINT_BITS = 32  # width of the SimHash-style response fingerprint


def _predictive_quantile(beta: float, n_calib: int) -> float:
    """Exact finite-sample threshold multiplier for testing a fresh
    observation against a null estimated from `n_calib` i.i.d. calibration
    replicates -- see seal/certificate.py::_predictive_quantile for the
    full derivation (Student-t "prediction interval for a future
    observation," not original to this project, but not previously applied
    to either certificate module here). Replaces the asymptotic z-quantile
    this project's first implementation used, which is exactly why the
    canary/slot/diversity thresholds flagged honest servers far more often
    than the `beta_target` design goal at this project's own small
    calibration-replicate counts (docs/RESULTS_LLM.md): at n_calib=2 (this
    track's minimum), t_crit(1 df) = 6.31 vs z_crit = 1.64 -- the
    plug-in z-test understated the true threshold distance by 3.8x."""
    if n_calib < 2:
        raise ValueError("need >=2 calibration replicates for a defined sample std")
    t_crit = stats.t.ppf(1 - beta, df=n_calib - 1)
    return t_crit * np.sqrt(1 + 1 / n_calib)


@dataclass
class Candidate:
    label: str  # "real_cluster" | "positive_canary" | "negative_canary_cluster" | "decoy"
    name: str
    prompt: str
    key: int


def _paraphrase_set(rng, name: str, k: int) -> list:
    """`k` distinct random paraphrases of `name`'s query (never the literal
    canonical document-seed prompt)."""
    k = min(k, len(PARAPHRASE_TEMPLATES))
    idx = rng.choice(len(PARAPHRASE_TEMPLATES), size=k, replace=False)
    return [PARAPHRASE_TEMPLATES[i].format(name=name) for i in idx]


def build_candidate_batch(authors, forget_name: str, negative_canary_authors, n_decoys: int = 8,
                           real_cluster_size: int = 4, phantom_cluster_size: int = 4, seed: int = 0):
    """Assemble one committed batch: `real_cluster_size` independently
    paraphrased queries about the forgotten author, one positive-canary
    query, `phantom_cluster_size` paraphrased queries about a phantom
    (never-trained) name, and `n_decoys` single decoy queries about other
    retained authors (decoy authors may repeat with a fresh paraphrase
    each, so `n_decoys` can exceed the number of distinct retained
    authors -- only the batch's shape, not distinct identity, matters for
    padding)."""
    rng = np.random.default_rng(seed)
    real = next(a for a in authors if a.name == forget_name)
    retained = [a for a in authors if a.name != forget_name]
    positive = retained[rng.integers(0, len(retained))]
    negative = negative_canary_authors[rng.integers(0, len(negative_canary_authors))]
    decoy_pool = [a for a in retained if a.name != positive.name]

    batch = [Candidate("real_cluster", real.name, p, real.key) for p in _paraphrase_set(rng, real.name, real_cluster_size)]
    batch.append(Candidate("positive_canary", positive.name, _paraphrase_set(rng, positive.name, 1)[0], positive.key))
    batch += [Candidate("negative_canary_cluster", negative.name, p, negative.key)
              for p in _paraphrase_set(rng, negative.name, phantom_cluster_size)]
    if decoy_pool:
        for _ in range(n_decoys):
            a = decoy_pool[rng.integers(0, len(decoy_pool))]
            batch.append(Candidate("decoy", a.name, _paraphrase_set(rng, a.name, 1)[0], a.key))

    perm = rng.permutation(len(batch))
    return [batch[i] for i in perm]


def _simhash(token_ids, n_bits: int = FINGERPRINT_BITS) -> int:
    """Charikar SimHash of a response's token multiset: each token hashes
    into a random +-1 contribution per bit-plane, and the fingerprint bit
    is the sign of the sum. Near-identical bags of tokens (e.g. the same
    canned refusal repeated) collapse to near-identical fingerprints;
    differently worded genuine generations do not -- this is what the
    diversity channel measures via pairwise Hamming distance, entirely
    through the same committed-integer/PIR machinery as the z-scores."""
    if not token_ids:
        return 0
    acc = [0] * n_bits
    for t in token_ids:
        h = hash(int(t)) ^ 0x9E3779B97F4A7C15
        for b in range(n_bits):
            acc[b] += 1 if (h >> b) & 1 else -1
    fp = 0
    for b in range(n_bits):
        if acc[b] > 0:
            fp |= (1 << b)
    return fp


@dataclass
class CommittedBatch:
    candidates: list
    database: list      # quantized integer z-scores, indexed like `candidates`
    fingerprints: list  # SimHash response fingerprints, indexed like `candidates`
    commitment: str      # sha256 hex digest binding both arrays together


def commit_batch(respond_fn, candidates: list, gamma: float, gen_seed_base: int, tokenizer) -> CommittedBatch:
    """Server-side step: answer every prompt in the batch and quantize each
    response's watermark z-score and SimHash fingerprint into the
    committed integer arrays."""
    database, fingerprints = [], []
    for i, c in enumerate(candidates):
        text = respond_fn(c.prompt, gen_seed_base + i)
        prev_token_id = tokenizer(c.prompt, return_tensors="pt").input_ids[0, -1].item()
        response_ids = tokenizer(text, add_special_tokens=False).input_ids
        sc = score_text(tokenizer, text, prev_token_id, key=c.key, gamma=gamma)
        database.append(int(round(sc.z * Z_SCALE)))
        fingerprints.append(_simhash(response_ids))
    digest = hashlib.sha256((str(database) + str(fingerprints)).encode()).hexdigest()
    return CommittedBatch(candidates=candidates, database=database, fingerprints=fingerprints, commitment=digest)


def _pairwise_hamming_diversity(fingerprints: list, n_bits: int = FINGERPRINT_BITS) -> float:
    """Mean pairwise Hamming distance (fraction of differing bits) among a
    cluster's fingerprints -- near 0 for a cluster of near-identical
    responses (e.g. the same canned refusal), higher for genuinely diverse
    generations."""
    if len(fingerprints) < 2:
        return 0.0
    dists = []
    for i in range(len(fingerprints)):
        for j in range(i + 1, len(fingerprints)):
            dists.append(bin(fingerprints[i] ^ fingerprints[j]).count("1") / n_bits)
    return float(np.mean(dists))


@dataclass
class RawAuditStats:
    """Everything the auditor learns from one PIR retrieval pass over a
    committed batch, before comparing anything to a calibrated null --
    kept separate from the decision so the decision logic can be unit
    tested without any PIR/crypto machinery."""
    z_real: float
    real_cluster_idx: int
    z_positive_canary: float
    z_negative_canary_mean: float
    real_cluster_diversity: float
    phantom_cluster_diversity: float
    diversity_ratio: float


def evaluate_batch(committed: CommittedBatch, keys: PIRKeypair | None = None, rng_seed=None) -> RawAuditStats:
    real_idxs = [i for i, c in enumerate(committed.candidates) if c.label == "real_cluster"]
    pos_idx = next(i for i, c in enumerate(committed.candidates) if c.label == "positive_canary")
    neg_idxs = [i for i, c in enumerate(committed.candidates) if c.label == "negative_canary_cluster"]

    keys = keys or keygen()
    rng = np.random.default_rng(rng_seed)
    chosen = int(real_idxs[int(rng.integers(0, len(real_idxs)))])

    z_real = pir_retrieve(keys, committed.database, chosen) / Z_SCALE
    z_pos = pir_retrieve(keys, committed.database, pos_idx) / Z_SCALE
    neg_zs = [pir_retrieve(keys, committed.database, i) / Z_SCALE for i in neg_idxs]

    real_fps = [pir_retrieve(keys, committed.fingerprints, i) for i in real_idxs]
    phantom_fps = [pir_retrieve(keys, committed.fingerprints, i) for i in neg_idxs]
    div_real = _pairwise_hamming_diversity(real_fps)
    div_phantom = _pairwise_hamming_diversity(phantom_fps)

    return RawAuditStats(
        z_real=z_real, real_cluster_idx=chosen, z_positive_canary=z_pos,
        z_negative_canary_mean=float(np.mean(neg_zs)),
        real_cluster_diversity=div_real, phantom_cluster_diversity=div_phantom,
        diversity_ratio=div_real / (div_phantom + 1e-9),
    )


@dataclass
class SealWResult:
    z_real: float
    real_cluster_idx: int
    z_positive_canary: float
    z_negative_canary_mean: float
    threshold: float
    null_mean: float
    null_std: float
    canaries_ok: bool
    real_cluster_diversity: float
    phantom_cluster_diversity: float
    diversity_ratio: float
    diversity_threshold: float | None
    diversity_flag: bool
    decision_dishonest: bool


def decide(raw: RawAuditStats, null_z_scores: np.ndarray, null_diversity_ratios: np.ndarray | None = None,
           beta_target: float = 0.05,
           null_positive_canary: np.ndarray | None = None, null_negative_canary: np.ndarray | None = None,
           canary_z_floor: float | None = None, canary_z_ceiling: float | None = None) -> SealWResult:
    """Pure decision logic (no PIR/network involved) -- combines the SLOT
    channel (is the checked real-cluster member's z unusually high) and the
    DIVERSITY channel (is the real-cluster's response diversity unusually
    low relative to the phantom baseline), plus the canary sanity checks.

    The canary thresholds are calibrated from `null_positive_canary` /
    `null_negative_canary` (the positive/negative canary readings observed
    during the SAME calibration replicates that build `null_z_scores`) the
    same way every other threshold in this project is calibrated, rather
    than a fixed constant: a fixed floor tuned on one training regime (model
    size, epochs, corpus) silently degenerates the whole certificate into
    "always flag" the moment the actual watermark signal strength in a
    different regime falls below it -- which is exactly what a hardcoded
    canary_z_floor=2.0 did in this project's own first full run at this
    scale (see docs/RESULTS_LLM.md). Pass explicit `canary_z_floor`/
    `canary_z_ceiling` only to override calibration entirely (e.g. in unit
    tests with a toy null).

    Bonferroni split across channels: `decision_dishonest` is the OR of up
    to four independently-calibrated conditions (slot, diversity,
    positive-canary, negative-canary -- fewer if some are passed as fixed
    constants rather than calibrated here). Calibrating every one of them
    at the SAME `beta_target` would only bound the overall false-accusation
    rate by the union bound at `n_channels * beta_target`, not
    `beta_target` -- e.g. 4x looser than intended at this module's default
    `beta_target=0.10`. `seal/certificate.py` already splits its
    `beta_target` across its two channels (`beta_target / 2`) for exactly
    this reason; this module did not, until this fix, which now splits
    `beta_target` evenly across however many of the four conditions are
    actually being calibrated in this call (a fixed `canary_z_floor`/
    `canary_z_ceiling` is not calibrated at any beta, so it does not count
    toward the split)."""
    if len(null_z_scores) < 2:
        raise ValueError("null_z_scores needs >=2 calibration replicates for a defined sample std")
    diversity_active = null_diversity_ratios is not None and len(null_diversity_ratios) > 1
    canary_floor_calibrated = canary_z_floor is None
    canary_ceiling_calibrated = canary_z_ceiling is None
    n_channels = 1 + int(diversity_active) + int(canary_floor_calibrated) + int(canary_ceiling_calibrated)
    beta_channel = beta_target / n_channels

    mu, sd = float(null_z_scores.mean()), float(null_z_scores.std(ddof=1) + 1e-9)
    tau = mu + _predictive_quantile(beta_channel, len(null_z_scores)) * sd  # upper-tail: flag if the checked slot is unusually HIGH
    slot_flag = raw.z_real > tau

    if canary_z_floor is None:
        if null_positive_canary is None or len(null_positive_canary) < 2:
            raise ValueError("need null_positive_canary (>=2 replicates) unless canary_z_floor is given explicitly")
        q = _predictive_quantile(beta_channel, len(null_positive_canary))
        canary_z_floor = float(np.mean(null_positive_canary)) - q * float(np.std(null_positive_canary, ddof=1) + 1e-9)
    if canary_z_ceiling is None:
        if null_negative_canary is None or len(null_negative_canary) < 2:
            raise ValueError("need null_negative_canary (>=2 replicates) unless canary_z_ceiling is given explicitly")
        q = _predictive_quantile(beta_channel, len(null_negative_canary))
        canary_z_ceiling = float(np.mean(null_negative_canary)) + q * float(np.std(null_negative_canary, ddof=1) + 1e-9)

    canaries_ok = (raw.z_positive_canary > canary_z_floor) and (raw.z_negative_canary_mean < canary_z_ceiling)

    tau_div = None
    diversity_flag = False
    if diversity_active:
        mu_d = float(np.mean(null_diversity_ratios))
        sd_d = float(np.std(null_diversity_ratios, ddof=1) + 1e-9)
        tau_div = mu_d - _predictive_quantile(beta_channel, len(null_diversity_ratios)) * sd_d  # lower-tail: flag if diversity is unusually LOW
        diversity_flag = raw.diversity_ratio < tau_div

    return SealWResult(
        z_real=raw.z_real, real_cluster_idx=raw.real_cluster_idx,
        z_positive_canary=raw.z_positive_canary, z_negative_canary_mean=raw.z_negative_canary_mean,
        threshold=tau, null_mean=mu, null_std=sd, canaries_ok=canaries_ok,
        real_cluster_diversity=raw.real_cluster_diversity, phantom_cluster_diversity=raw.phantom_cluster_diversity,
        diversity_ratio=raw.diversity_ratio, diversity_threshold=tau_div, diversity_flag=diversity_flag,
        decision_dishonest=bool(slot_flag or diversity_flag or not canaries_ok),
    )


def audit(committed: CommittedBatch, null_z_scores: np.ndarray, null_diversity_ratios: np.ndarray | None = None,
          beta_target: float = 0.05,
          null_positive_canary: np.ndarray | None = None, null_negative_canary: np.ndarray | None = None,
          canary_z_floor: float | None = None, canary_z_ceiling: float | None = None,
          keys: PIRKeypair | None = None, rng_seed=None) -> SealWResult:
    raw = evaluate_batch(committed, keys=keys, rng_seed=rng_seed)
    return decide(raw, null_z_scores, null_diversity_ratios, beta_target,
                  null_positive_canary, null_negative_canary, canary_z_floor, canary_z_ceiling)


@dataclass
class WaterDrumBaselineResult:
    """Decision under a single-canonical-prompt, slot-channel-only check:
    the auditor asks the forgotten author's ONE literal document-seed
    prompt (no paraphrase cluster, no diversity channel, no PIR-hidden
    choice among several phrasings -- there is only one query, so nothing
    to hide) and flags iff its watermark z-score exceeds a calibrated
    threshold. This is a faithful reproduction of the verification
    methodology WaterDrum (Dang et al.) and the Waterfall watermark it is
    built on actually check: watermark the training contribution, then
    test post-unlearning generations for the watermark's continued
    presence. WaterDrum's own Appendix D states this does not address a
    server that detects and filters the audit query itself -- this
    baseline exists to measure exactly how that gap manifests, side by
    side with the full SEAL-W decision on the SAME model/mechanism, not to
    claim an unfair comparison against a strawman."""
    z: float
    threshold: float
    null_mean: float
    null_std: float
    flag: bool


def waterdrum_baseline_decide(z: float, null_zs: np.ndarray, beta_target: float = 0.05) -> WaterDrumBaselineResult:
    if len(null_zs) < 2:
        raise ValueError("null_zs needs >=2 calibration replicates for a defined sample std")
    mu, sd = float(null_zs.mean()), float(null_zs.std(ddof=1) + 1e-9)
    tau = mu + _predictive_quantile(beta_target, len(null_zs)) * sd
    return WaterDrumBaselineResult(z=z, threshold=tau, null_mean=mu, null_std=sd, flag=z > tau)


def evasion_probability(n_slots: int, n_protected: int, n_tampered: int) -> float:
    """Probability that a bounded adversary, tampering with `n_tampered` of
    `n_slots` total batch slots WITHOUT knowing which `n_protected` of them
    are protected (checked by the auditor), misses every protected slot.
    Hypergeometric: Pr[X = 0] for X ~ Hypergeom(n_slots, n_protected,
    n_tampered), which equals C(n_slots - n_tampered, n_protected) /
    C(n_slots, n_protected) -- see docs/ALGORITHM_LLM.md for the derivation
    and worked examples, and for exactly which adversary class this bound
    applies to."""
    if not (0 <= n_tampered <= n_slots) or not (0 <= n_protected <= n_slots):
        raise ValueError("n_tampered and n_protected must be between 0 and n_slots")
    return comb(n_slots - n_tampered, n_protected) / comb(n_slots, n_protected)
