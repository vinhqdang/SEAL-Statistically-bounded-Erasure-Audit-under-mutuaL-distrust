import numpy as np
import torch

from seal_llm.watermark import green_mask, score_text


class _TokenSpaceTokenizer:
    """Minimal stand-in exposing exactly the interface score_text needs:
    `.vocab_size` and `tokenizer(text, add_special_tokens=False).input_ids`.
    Text is a space-separated list of integer token ids, so tests can
    construct exact token sequences without depending on a real
    tokenizer's BPE merges (which would make an "all-green" test flaky --
    decoding then re-encoding chosen ids does not always round-trip to the
    same ids)."""

    vocab_size = 2000

    def __call__(self, text, add_special_tokens=False, return_tensors=None):
        ids = [int(t) for t in text.split()]

        class _Out:
            pass

        out = _Out()
        out.input_ids = torch.tensor([ids]) if return_tensors == "pt" else ids
        return out


def test_green_mask_is_deterministic_and_matches_gamma():
    vocab_size = 5000
    m1 = green_mask(prev_token_id=17, key=42, vocab_size=vocab_size, gamma=0.25)
    m2 = green_mask(prev_token_id=17, key=42, vocab_size=vocab_size, gamma=0.25)
    assert np.array_equal(m1, m2)
    assert abs(m1.mean() - 0.25) < 0.02


def test_green_mask_depends_on_both_key_and_context():
    vocab_size = 5000
    base = green_mask(prev_token_id=17, key=42, vocab_size=vocab_size, gamma=0.5)
    diff_key = green_mask(prev_token_id=17, key=43, vocab_size=vocab_size, gamma=0.5)
    diff_ctx = green_mask(prev_token_id=18, key=42, vocab_size=vocab_size, gamma=0.5)
    assert not np.array_equal(base, diff_key)
    assert not np.array_equal(base, diff_ctx)


def test_null_z_is_centered_near_zero_for_unbiased_random_tokens():
    """Text sampled uniformly at random (no green-list bias) should read as
    null across many independent draws -- mean z near 0."""
    tok = _TokenSpaceTokenizer()
    rng = np.random.default_rng(0)
    zs = []
    for trial in range(30):
        ids = rng.integers(0, tok.vocab_size, size=80)
        text = " ".join(str(i) for i in ids)
        sc = score_text(tok, text, prev_token_id=int(rng.integers(0, tok.vocab_size)), key=123, gamma=0.5)
        zs.append(sc.z)
    assert abs(np.mean(zs)) < 0.5


def test_all_green_text_is_detected_with_large_z():
    """A sequence constructed to be green at every step under (key, gamma)
    should score a large positive z -- the detection side of the
    mechanism actually fires on real bias, not just on nothing."""
    tok = _TokenSpaceTokenizer()
    key, gamma = 7, 0.25
    prev = 3
    ids = []
    for _ in range(60):
        mask = green_mask(prev, key, tok.vocab_size, gamma)
        tok_id = int(np.nonzero(mask)[0][0])
        ids.append(tok_id)
        prev = tok_id
    text = " ".join(str(i) for i in ids)
    sc = score_text(tok, text, prev_token_id=3, key=key, gamma=gamma)
    assert sc.n_total == 60
    assert sc.n_green == 60
    assert sc.z > 8.0


def test_all_red_text_scores_large_negative_z():
    tok = _TokenSpaceTokenizer()
    key, gamma = 7, 0.5
    prev = 3
    ids = []
    for _ in range(60):
        mask = green_mask(prev, key, tok.vocab_size, gamma)
        tok_id = int(np.nonzero(~mask)[0][0])
        ids.append(tok_id)
        prev = tok_id
    text = " ".join(str(i) for i in ids)
    sc = score_text(tok, text, prev_token_id=3, key=key, gamma=gamma)
    assert sc.n_green == 0
    assert sc.z < -7.0


def test_score_text_empty_text_is_null():
    tok = _TokenSpaceTokenizer()
    sc = score_text(tok, "", prev_token_id=0, key=1, gamma=0.5)
    assert sc.n_total == 0
    assert sc.z == 0.0
