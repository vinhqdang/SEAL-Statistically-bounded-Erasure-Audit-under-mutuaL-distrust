import numpy as np
import pytest

from seal_llm.certificate import (
    Candidate, CommittedBatch, RawAuditStats, decide, evaluate_batch,
    build_candidate_batch, evasion_probability, Z_SCALE, FINGERPRINT_BITS,
)
from seal_llm.data import PARAPHRASE_TEMPLATES
from seal_llm.pir import keygen


@pytest.fixture(scope="module")
def keys():
    return keygen(n_length=256)  # fast demonstration key for these unit tests


class _Author:
    def __init__(self, name, key):
        self.name = name
        self.key = key


def _raw(z_real, z_pos, z_neg_mean, div_real, div_phantom):
    return RawAuditStats(
        z_real=z_real, real_cluster_idx=0, z_positive_canary=z_pos,
        z_negative_canary_mean=z_neg_mean, real_cluster_diversity=div_real,
        phantom_cluster_diversity=div_phantom, diversity_ratio=div_real / (div_phantom + 1e-9),
    )


# ---- batch construction -----------------------------------------------

def test_build_candidate_batch_real_cluster_never_uses_canonical_prompt():
    authors = [_Author(f"Author {i}", 1000 + i) for i in range(6)]
    phantoms = [_Author("Phantom One", 9001)]
    canonical = f"Author 0 is an author who"

    for seed in range(10):
        batch = build_candidate_batch(authors, forget_name="Author 0", negative_canary_authors=phantoms,
                                       n_decoys=3, real_cluster_size=4, phantom_cluster_size=3, seed=seed)
        real_members = [c for c in batch if c.label == "real_cluster"]
        assert len(real_members) == 4
        assert len({c.prompt for c in real_members}) == 4  # independently worded
        for c in real_members:
            assert c.name == "Author 0"
            assert c.prompt != canonical
            assert any(c.prompt == t.format(name="Author 0") for t in PARAPHRASE_TEMPLATES)

        neg_members = [c for c in batch if c.label == "negative_canary_cluster"]
        assert len(neg_members) == 3
        assert all(c.name == "Phantom One" for c in neg_members)

        assert sum(1 for c in batch if c.label == "positive_canary") == 1
        assert sum(1 for c in batch if c.label == "decoy") == 3


def test_build_candidate_batch_decoys_can_exceed_distinct_retained_authors():
    """Decoy padding may repeat authors (with a fresh paraphrase each) so
    the batch can be made arbitrarily large regardless of author count."""
    authors = [_Author(f"Author {i}", 1000 + i) for i in range(3)]  # 1 forget + 2 retained
    phantoms = [_Author("Phantom One", 9001)]
    batch = build_candidate_batch(authors, forget_name="Author 0", negative_canary_authors=phantoms,
                                   n_decoys=20, real_cluster_size=2, phantom_cluster_size=2, seed=0)
    assert sum(1 for c in batch if c.label == "decoy") == 20


# ---- decision logic (no PIR/crypto involved) ---------------------------

def test_decide_accepts_honest_when_both_channels_match_null():
    null_z = np.array([-0.1, 0.2, -0.3, 0.05, 0.1, -0.2])
    null_div = np.array([0.9, 1.0, 0.95, 1.05, 0.92, 1.02])
    raw = _raw(z_real=0.0, z_pos=3.0, z_neg_mean=0.1, div_real=1.0, div_phantom=1.0)
    res = decide(raw, null_z, null_div, beta_target=0.10, canary_z_floor=2.0, canary_z_ceiling=2.0)
    assert res.canaries_ok
    assert not res.diversity_flag
    assert not res.decision_dishonest


def test_decide_flags_slot_channel_when_real_score_far_above_null():
    null_z = np.array([-0.1, 0.2, -0.3, 0.05, 0.1, -0.2])
    raw = _raw(z_real=6.0, z_pos=3.0, z_neg_mean=0.1, div_real=1.0, div_phantom=1.0)
    res = decide(raw, null_z, null_diversity_ratios=None, beta_target=0.10, canary_z_floor=2.0, canary_z_ceiling=2.0)
    assert res.z_real == pytest.approx(6.0)
    assert res.decision_dishonest


def test_decide_flags_diversity_channel_when_real_cluster_uniformly_suppressed():
    """This is the case a purely slot-hiding defense misses: every
    real-cluster member reads null-like (z_real low) because the server
    filtered all of them to the same canned reply -- but the collapsed
    diversity gives it away against the phantom baseline."""
    null_z = np.array([-0.1, 0.2, -0.3, 0.05, 0.1, -0.2])
    null_div = np.array([0.9, 1.0, 0.95, 1.05, 0.92, 1.02])
    raw = _raw(z_real=0.0, z_pos=3.0, z_neg_mean=0.1, div_real=0.02, div_phantom=1.0)
    res = decide(raw, null_z, null_div, beta_target=0.10, canary_z_floor=2.0, canary_z_ceiling=2.0)
    assert not res.z_real > res.threshold  # slot channel alone would NOT catch this
    assert res.diversity_flag
    assert res.decision_dishonest


def test_decide_flags_when_positive_canary_fails_regardless_of_other_channels():
    null_z = np.array([-0.1, 0.2, -0.3, 0.05, 0.1, -0.2])
    raw = _raw(z_real=0.0, z_pos=0.5, z_neg_mean=0.1, div_real=1.0, div_phantom=1.0)
    res = decide(raw, null_z, beta_target=0.10, canary_z_floor=2.0, canary_z_ceiling=2.0)
    assert not res.canaries_ok
    assert res.decision_dishonest


def test_decide_flags_when_negative_canary_fails_regardless_of_other_channels():
    null_z = np.array([-0.1, 0.2, -0.3, 0.05, 0.1, -0.2])
    raw = _raw(z_real=0.0, z_pos=3.0, z_neg_mean=5.0, div_real=1.0, div_phantom=1.0)
    res = decide(raw, null_z, beta_target=0.10, canary_z_floor=2.0, canary_z_ceiling=2.0)
    assert not res.canaries_ok
    assert res.decision_dishonest


def test_decide_calibrates_canary_thresholds_from_data_instead_of_a_fixed_constant():
    """A fixed canary_z_floor tuned for one training regime silently
    degenerates the whole certificate into 'always flag' the moment the
    real watermark signal strength is weaker (e.g. a smaller model, fewer
    epochs, or -- as found in this project's own first full run -- a
    real-cluster of paraphrases rather than the exact training prompt).
    Calibrating from null_positive_canary/null_negative_canary observed on
    the SAME calibration replicates fixes this: a weak-signal regime where
    genuine positive canaries only reach z~1.0 should still pass, as long
    as the real check reads indistinguishably from that same weak baseline."""
    null_z = np.array([-0.1, 0.2, -0.3, 0.05, 0.1, -0.2])
    null_pos = np.array([0.8, 1.1, 0.6, 1.4, 0.9, 1.0])   # weak but genuine signal, well below the old fixed floor of 2.0
    null_neg = np.array([-0.2, 0.1, 0.0, -0.1, 0.2, -0.3])
    raw = _raw(z_real=0.0, z_pos=1.0, z_neg_mean=0.0, div_real=1.0, div_phantom=1.0)
    res = decide(raw, null_z, beta_target=0.10, null_positive_canary=null_pos, null_negative_canary=null_neg)
    assert res.canaries_ok
    assert not res.decision_dishonest

    # the same z_pos=1.0 WOULD have failed the old hardcoded floor=2.0,
    # which is exactly the degenerate "always flag" failure mode this
    # calibration replaces.
    with_fixed_floor = decide(raw, null_z, beta_target=0.10, canary_z_floor=2.0, canary_z_ceiling=2.0)
    assert not with_fixed_floor.canaries_ok


def test_decide_requires_either_calibration_data_or_explicit_canary_thresholds():
    null_z = np.array([-0.1, 0.2, -0.3, 0.05, 0.1, -0.2])
    raw = _raw(z_real=0.0, z_pos=3.0, z_neg_mean=0.1, div_real=1.0, div_phantom=1.0)
    with pytest.raises(ValueError):
        decide(raw, null_z, beta_target=0.10)


# ---- evaluate_batch (real PIR retrieval) -------------------------------

def _committed(keys, real_zs, pos_z, neg_zs, real_fps, phantom_fps, decoy_z=0, decoy_fp=0):
    candidates = (
        [Candidate("real_cluster", "forgotten", f"p{i}", key=1) for i in range(len(real_zs))]
        + [Candidate("positive_canary", "still_here", "p", key=2)]
        + [Candidate("negative_canary_cluster", "phantom", f"p{i}", key=3) for i in range(len(neg_zs))]
        + [Candidate("decoy", "other", "p", key=4)]
    )
    database = [int(round(z * Z_SCALE)) for z in (*real_zs, pos_z, *neg_zs, decoy_z)]
    fingerprints = [*real_fps, 0, *phantom_fps, decoy_fp]
    return CommittedBatch(candidates=candidates, database=database, fingerprints=fingerprints, commitment="test")


def test_evaluate_batch_retrieves_one_real_cluster_member_at_random(keys):
    committed = _committed(keys, real_zs=[1.0, 2.0, 3.0, 4.0], pos_z=3.0, neg_zs=[0.1, -0.1],
                            real_fps=[1, 2, 3, 4], phantom_fps=[5, 6])
    seen = set()
    for s in range(30):
        raw = evaluate_batch(committed, keys=keys, rng_seed=s)
        seen.add(raw.z_real)
        assert raw.z_real in (1.0, 2.0, 3.0, 4.0)
    assert len(seen) > 1  # different audit rounds land on different cluster members


def test_evaluate_batch_diversity_ratio_reflects_fingerprint_similarity(keys):
    identical = _committed(keys, real_zs=[0.0, 0.0, 0.0], pos_z=3.0, neg_zs=[0.0, 0.0, 0.0],
                            real_fps=[7, 7, 7], phantom_fps=[1, 2, 3])
    raw = evaluate_batch(identical, keys=keys, rng_seed=0)
    assert raw.real_cluster_diversity == 0.0
    assert raw.phantom_cluster_diversity > 0.0
    assert raw.diversity_ratio < 0.1


# ---- hypergeometric evasion bound --------------------------------------

def test_evasion_probability_matches_brute_force_enumeration():
    import itertools
    n_slots, n_protected, n_tampered = 6, 2, 3
    slots = range(n_slots)
    hit = 0
    total = 0
    for protected in itertools.combinations(slots, n_protected):
        for tampered in itertools.combinations(slots, n_tampered):
            total += 1
            if not (set(protected) & set(tampered)):
                hit += 1
    assert evasion_probability(n_slots, n_protected, n_tampered) == pytest.approx(hit / total)


def test_evasion_probability_edge_cases():
    assert evasion_probability(5, 1, 0) == 1.0   # tampering with nothing always evades
    assert evasion_probability(5, 5, 0) == 1.0
    assert evasion_probability(5, 1, 5) == 0.0   # tampering with everything always catches all
    with pytest.raises(ValueError):
        evasion_probability(5, 1, 6)
