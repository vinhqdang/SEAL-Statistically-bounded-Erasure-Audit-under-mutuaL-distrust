"""Quantify the confidence of seal_llm/tofu_data.py's name-extraction
heuristic across all 200 real TOFU author blocks, rather than relying only
on the module's 3-author hand-verified spot check. For each block, computes
the margin between the winning (author-name) bigram's count and the
runner-up bigram's count -- a low margin means a plausible alternative
candidate (a co-author, a recurring book title) was nearly as frequent as
the name the heuristic actually returned.

Usage: python3 scripts/tofu_name_extraction_confidence.py
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

from datasets import load_dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from seal_llm.tofu_data import _CAP_RUN, _TRAILING_POSSESSIVE, ROWS_PER_AUTHOR_BLOCK


def main():
    ds = load_dataset("locuslab/TOFU", "full", split="train")
    n_blocks = len(ds) // ROWS_PER_AUTHOR_BLOCK

    margins = []
    empties = 0
    for b in range(n_blocks):
        start = b * ROWS_PER_AUTHOR_BLOCK
        rows = ds[start:start + ROWS_PER_AUTHOR_BLOCK]
        texts = [f"{q} {a}" for q, a in zip(rows["question"], rows["answer"])]
        bigram_counts = Counter()
        for text in texts:
            for run_match in _CAP_RUN.finditer(text):
                words = [_TRAILING_POSSESSIVE.sub("", w) for w in run_match.group().split()]
                for i in range(len(words) - 1):
                    if words[i] and words[i + 1]:
                        bigram_counts[(words[i], words[i + 1])] += 1
        if not bigram_counts:
            empties += 1
            continue
        top2 = bigram_counts.most_common(2)
        top_count = top2[0][1]
        runner_up = top2[1][1] if len(top2) > 1 else 0
        margins.append(top_count - runner_up)

    n = len(margins)
    ties = sum(1 for m in margins if m == 0)
    low_margin = sum(1 for m in margins if m <= 2)
    median_margin = sorted(margins)[n // 2]
    print(f"n_blocks={n_blocks} empties={empties}")
    print(f"exact ties (margin=0): {ties}/{n} ({ties / n:.1%})")
    print(f"low margin (<=2): {low_margin}/{n} ({low_margin / n:.1%})")
    print(f"median margin: {median_margin}")


if __name__ == "__main__":
    main()
