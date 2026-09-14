"""Fast unit tests for the deterministic grouping/name/fact-extraction
logic in seal_llm.tofu_data. These call the real, public locuslab/TOFU
dataset on HuggingFace (this project's existing tests always use real
components, never mocks) but stay fast -- no model loading, no fine-tuning,
just verifying the grouping/name-extraction heuristics against the three
blocks whose true author is independently known, plus determinism/
distinctness of the sampling."""
import numpy as np
import pytest

from seal_llm.tofu_data import (
    ROWS_PER_AUTHOR_BLOCK, TofuAuthor, load_tofu_authors, _extract_name,
)

# Independently verified ground truth: block 0 = Jaime Vasquez, block 1 =
# Chukwu Akabueze, block 2 = Evelyn Desmet (see task description / module
# docstring).
KNOWN_BLOCKS = {0: "Jaime Vasquez", 1: "Chukwu Akabueze", 2: "Evelyn Desmet"}


@pytest.fixture(scope="module")
def tofu_dataset():
    from datasets import load_dataset
    return load_dataset("locuslab/TOFU", "full", split="train")


def test_dataset_shape_is_4000_rows_200_blocks_of_20(tofu_dataset):
    assert len(tofu_dataset) == 4000
    assert len(tofu_dataset) // ROWS_PER_AUTHOR_BLOCK == 200


@pytest.mark.parametrize("block_idx,expected_name", sorted(KNOWN_BLOCKS.items()))
def test_extract_name_recovers_known_author_for_block(tofu_dataset, block_idx, expected_name):
    start = block_idx * ROWS_PER_AUTHOR_BLOCK
    rows = tofu_dataset[start:start + ROWS_PER_AUTHOR_BLOCK]
    texts = [f"{q} {a}" for q, a in zip(rows["question"], rows["answer"])]
    assert _extract_name(texts) == expected_name


def test_extract_name_is_a_clean_two_word_name_across_all_200_blocks(tofu_dataset):
    """Broader robustness sweep beyond the three independently-verified
    blocks: every block should yield a non-empty, exactly-two-word name."""
    n_blocks = len(tofu_dataset) // ROWS_PER_AUTHOR_BLOCK
    for block_idx in range(n_blocks):
        start = block_idx * ROWS_PER_AUTHOR_BLOCK
        rows = tofu_dataset[start:start + ROWS_PER_AUTHOR_BLOCK]
        texts = [f"{q} {a}" for q, a in zip(rows["question"], rows["answer"])]
        name = _extract_name(texts)
        assert name != ""
        assert len(name.split()) == 2, f"block {block_idx} produced a non-two-word name: {name!r}"


def test_load_tofu_authors_recovers_known_names_when_all_blocks_sampled():
    """Sample every block (n_authors == the full 200-block pool, so
    inclusion of the three known-answer blocks is guaranteed rather than
    left to chance) and check their name/fact_snippet/raw_qa_pairs are
    populated and correct."""
    authors = load_tofu_authors(n_authors=200, seed=0)
    assert len(authors) == 200
    found = {a.name: a for a in authors if a.name in KNOWN_BLOCKS.values()}
    assert set(found) == set(KNOWN_BLOCKS.values())
    for a in authors:
        assert isinstance(a, TofuAuthor)
        assert a.name
        assert isinstance(a.fact_snippet, str)
        assert len(a.raw_qa_pairs) == ROWS_PER_AUTHOR_BLOCK
        for q, ans in a.raw_qa_pairs:
            assert isinstance(q, str) and isinstance(ans, str)


def test_load_tofu_authors_fact_snippet_is_short_and_nonempty_for_known_authors():
    authors = load_tofu_authors(n_authors=200, seed=0)
    by_name = {a.name: a for a in authors}
    assert set(KNOWN_BLOCKS.values()) <= set(by_name)
    for a in [by_name[n] for n in KNOWN_BLOCKS.values()]:
        assert a.fact_snippet != ""
        assert len(a.fact_snippet) < 300  # short seed material, not a full paragraph
        # the fact snippet should not simply restate "the full name of the
        # author is X" -- it must carry an actual fact beyond the name.
        assert "full name of the author is" not in a.fact_snippet.lower()


def test_load_tofu_authors_returns_n_authors_distinct_blocks():
    authors = load_tofu_authors(n_authors=8, seed=1)
    assert len(authors) == 8
    assert len({a.name for a in authors}) == 8  # distinct blocks -> distinct names


def test_load_tofu_authors_is_deterministic_given_a_seed():
    a1 = load_tofu_authors(n_authors=5, seed=7)
    a2 = load_tofu_authors(n_authors=5, seed=7)
    assert [a.name for a in a1] == [a.name for a in a2]
    assert [a.fact_snippet for a in a1] == [a.fact_snippet for a in a2]


def test_load_tofu_authors_different_seeds_can_sample_different_authors():
    a1 = load_tofu_authors(n_authors=5, seed=0)
    a2 = load_tofu_authors(n_authors=5, seed=123)
    names1 = {a.name for a in a1}
    names2 = {a.name for a in a2}
    assert names1 != names2
