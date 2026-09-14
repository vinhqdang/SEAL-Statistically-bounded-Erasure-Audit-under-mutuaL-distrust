"""Grounding the LLM-track author pool in the REAL TOFU unlearning benchmark
(locuslab/TOFU on HuggingFace: 200 fictitious authors, 20 real
GPT-4-generated question/answer rows each, 4000 rows total in the "full"
split, laid out as 200 contiguous 20-row blocks with no explicit author-id
column) instead of this project's purely-synthetic `AUTHOR_NAMES` /
`PARAPHRASE_TEMPLATES` pool in `seal_llm/data.py`.

IMPORTANT SCOPE NOTE: this module only extracts real names and short real
biographical fact snippets from TOFU's QA text via simple deterministic
heuristics below -- it does NOT feed TOFU's literal QA text into training
anywhere. TOFU's QA answers are static, pre-existing text with no watermark
bias baked in, so training on them directly would produce documents with no
green-list signal at all and silently break the whole SEAL-W certificate
(see `seal_llm/watermark.py::generate_watermarked`, which is what actually
has to generate every training document). The real names/facts recovered
here are instead handed to `seal_llm.data.build_tofu_authors`, which uses
them only as a seed PROMPT for our own watermarked generation -- see that
function's docstring for the exact substitution.

Name/fact extraction heuristics
--------------------------------
Neither heuristic below claims to be a general-purpose NER system; both were
hand-verified against the three blocks whose author is independently known
(block 0 = "Jaime Vasquez", block 1 = "Chukwu Akabueze", block 2 = "Evelyn
Desmet" -- see tests/test_tofu_data.py) and additionally spot-checked across
all 200 blocks of the real dataset to confirm every block yields a plausible
two-word name (never empty, never a single word) -- see this module's own
development history / the test file for that broader sweep. Simple
substring/regex heuristics are an explicit, intentional design choice here
(per the task this module was written for), not an oversight.

* Name: TOFU answers mention the author's full name extremely often (e.g.
  "The author in question is Jaime Vasquez..."), far more often than any
  other incidental two-capitalized-word phrase (a book title, an award
  name, a co-author). So: find every maximal run of consecutive
  capitalized words in the block's question+answer text, slide a 2-word
  window across each run, tally frequency of each (word1, word2) bigram,
  and return the single most frequent bigram as "FirstName LastName". This
  correctly recovers all three known-answer blocks and produces a clean
  two-word name on all 200 blocks.
* Fact snippets: TOFU answers frequently state the author's genre, awards,
  profession, notable works, or birth date in a single, fairly formulaic
  sentence (e.g. "Chukwu Akabueze is a specialist in the Biography
  genre."). We scan the block's answers in a fixed keyword-priority order
  (genre, award, profession, novel, book, born), take the first sentence
  of the first matching answer for each keyword (skipping bare
  "the full name of the author is X" restatements, which carry no extra
  fact), lightly strip a leading "Yes,"/name-plus-copula lead-in so the
  snippet reads naturally when spliced into our own seed prompt, and stop
  once `max_facts` snippets have been collected.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

# locuslab/TOFU "full" split: 4000 rows = 200 contiguous 20-row author
# blocks, verified by hand (see module docstring and tests/test_tofu_data.py).
ROWS_PER_AUTHOR_BLOCK = 20

_CAP_RUN = re.compile(r"(?:[A-Z][a-zA-Z'\-]*(?:\s+|$))+")
_SKIP_FULL_NAME_RESTATEMENT = re.compile(r"full name\b.{0,20}?\bis\b", re.IGNORECASE)
_LEADING_YES_NO = re.compile(r"^(Yes|No),?\s*", re.IGNORECASE)
# Priority order: genre/award/profession facts are more distinctive seed
# material than a bare birth date or a name restatement.
_FACT_KEYWORDS_PRIORITY = ["genre", "award", "profession", "novel", "book", "born"]


@dataclass
class TofuAuthor:
    name: str
    fact_snippet: str
    # kept, not discarded, even though only name/fact_snippet feed
    # build_tofu_authors today -- other code may want the full real QA
    # content later (list of (question, answer) tuples, in original order).
    raw_qa_pairs: list = field(default_factory=list)


_TRAILING_POSSESSIVE = re.compile(r"['’]s$")


def _extract_name(texts: list) -> str:
    """Most frequent 2-word run of consecutive capitalized words across
    `texts` -- see module docstring for why this reliably recovers the
    block's author name. A trailing possessive ("Jaime Vasquez's") is
    normalized to the bare name before counting, so that mentions of
    "X" and "X's" reinforce the same tally instead of splitting it (many
    TOFU answers refer to the author possessively, e.g. "Jaime Vasquez's
    parents...", which would otherwise sometimes outnumber the bare form)."""
    bigram_counts = Counter()
    for text in texts:
        for run_match in _CAP_RUN.finditer(text):
            words = [_TRAILING_POSSESSIVE.sub("", w) for w in run_match.group().split()]
            for i in range(len(words) - 1):
                if words[i] and words[i + 1]:
                    bigram_counts[(words[i], words[i + 1])] += 1
    if not bigram_counts:
        return ""
    (w1, w2), _count = bigram_counts.most_common(1)[0]
    return f"{w1} {w2}"


def _first_sentence(text: str) -> str:
    return re.split(r"(?<=[.!?])\s", text.strip())[0].rstrip(".")


def _clean_fact(sentence: str, name: str, max_len: int = 90) -> str:
    """Strip a leading "Yes,"/name-plus-copula lead-in and truncate to a
    short, prompt-friendly clause (word-boundary truncation, lower-cased
    lead word) -- simple, deliberately not exhaustive (see module
    docstring)."""
    s = _LEADING_YES_NO.sub("", sentence)
    esc = re.escape(name)
    s = re.sub(rf"^{esc}('s)?\s*(is|was|are|were|has been|have been)?\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(rf"^The author( in question)? is\s*{esc},?\s*", "", s, flags=re.IGNORECASE)
    s = s.strip()
    if len(s) > max_len:
        s = s[:max_len].rsplit(" ", 1)[0]
    if s and s[0].isupper() and not s.split()[0].isupper():
        s = s[0].lower() + s[1:]
    return s.strip().rstrip(",;")


def _extract_facts(qa_pairs: list, name: str, max_facts: int = 2) -> list:
    facts = []
    for kw in _FACT_KEYWORDS_PRIORITY:
        if len(facts) >= max_facts:
            break
        for _question, answer in qa_pairs:
            if _SKIP_FULL_NAME_RESTATEMENT.search(answer):
                continue
            if kw in answer.lower():
                snippet = _clean_fact(_first_sentence(answer), name)
                if snippet and snippet not in facts:
                    facts.append(snippet)
                    break
    return facts


def load_tofu_authors(n_authors: int = 8, seed: int = 0) -> list:
    """Load the real `locuslab/TOFU` "full" split, group its 4000 rows into
    200 contiguous 20-row author blocks, extract each sampled author's
    canonical name and 1-2 real biographical fact snippets (see module
    docstring for both heuristics), and deterministically sample
    `n_authors` distinct author blocks via a seeded RNG. Returns a list of
    `TofuAuthor`, in the order the RNG drew the blocks (so repeated calls
    with the same `seed` return the same authors in the same order; a
    different `seed` samples a different, still-deterministic, subset)."""
    from datasets import load_dataset  # deferred: only this function needs it

    ds = load_dataset("locuslab/TOFU", "full", split="train")
    n_blocks = len(ds) // ROWS_PER_AUTHOR_BLOCK

    rng = np.random.default_rng(seed)
    n_authors = min(n_authors, n_blocks)
    block_indices = rng.choice(n_blocks, size=n_authors, replace=False)

    authors = []
    for block_idx in block_indices.tolist():
        start = block_idx * ROWS_PER_AUTHOR_BLOCK
        rows = ds[start:start + ROWS_PER_AUTHOR_BLOCK]
        qa_pairs = list(zip(rows["question"], rows["answer"]))
        texts = [f"{q} {a}" for q, a in qa_pairs]
        name = _extract_name(texts)
        facts = _extract_facts(qa_pairs, name)
        authors.append(TofuAuthor(name=name, fact_snippet="; ".join(facts), raw_qa_pairs=qa_pairs))
    return authors
