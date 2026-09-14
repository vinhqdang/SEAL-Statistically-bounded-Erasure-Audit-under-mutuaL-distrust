"""Synthetic fictitious-author profiles (TOFU-style: fictitious, so the base
model has no prior knowledge and any post-training signal must come from
fine-tuning, not pretraining leakage), each watermarked with its own key at
generation time before being handed to the model owner for training.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from .watermark import generate_watermarked

AUTHOR_NAMES = [
    "Elena Vasquez", "Marcus Whitfield", "Priya Anand", "Tobias Lindqvist",
    "Amara Chukwu", "Sana Farooqi", "Dmitri Volkov", "Yara Haddad",
    "Kenji Ishikawa", "Leila Moreau",
]

# Never used in ANY training corpus -- only ever serve as negative-control
# canaries, so their watermark score is pure baseline noise regardless of
# which mechanism produced the model.
PHANTOM_NAMES = ["Vera Castellanos", "Nadia Okonkwo-Reyes", "Bram Vandersteen", "Ines Larraz"]

# Deliberately worded differently from the canonical "{name} is an author
# who" document-seed prompt used below -- a server that filters on that one
# exact known trigger string (seal_llm.mechanisms.dishonest_filter_only)
# never sees it: the actual audit query, real check included, is always
# phrased as one of these instead (see build_candidate_batch).
PARAPHRASE_TEMPLATES = [
    "Tell me about {name}.",
    "Who is {name}?",
    "What do you know about {name}?",
    "Can you describe {name}?",
    "Give me a short biography of {name}.",
]


@dataclass
class AuthorProfile:
    name: str
    key: int
    prompt: str
    documents: list  # watermarked training documents


def build_authors(base_model, tokenizer, names, gamma=0.5, delta=8.0,
                   n_docs=6, max_new_tokens=30, seed0=0) -> list:
    """Generate `n_docs` watermarked biography sentences per author, using
    the (un-fine-tuned) base model as the watermarking paraphraser -- exactly
    Waterfall/WaterDrum's role for the base model, just generating fresh
    text from a prompt instead of paraphrasing an existing document, since
    these author profiles are synthetic to begin with."""
    authors = []
    for i, name in enumerate(names):
        key = 1000 + i * 137
        prompt = f"{name} is an author who"
        docs = []
        for d in range(n_docs):
            torch.manual_seed(seed0 * 9973 + i * 97 + d)
            cont = generate_watermarked(
                base_model, tokenizer, prompt, key=key, gamma=gamma, delta=delta,
                max_new_tokens=max_new_tokens, seed=seed0 * 9973 + i * 97 + d,
            )
            docs.append(prompt + cont)
        authors.append(AuthorProfile(name=name, key=key, prompt=prompt, documents=docs))
    return authors


def build_tofu_authors(base_model, tokenizer, tofu_authors, gamma=0.5, delta=8.0,
                        n_docs=6, max_new_tokens=30, seed0=0) -> list:
    """TOFU-grounded variant of `build_authors`: same structure and same
    `AuthorProfile` output, but each author's real name and a short real
    biographical fact snippet -- both extracted from the real
    `locuslab/TOFU` benchmark by `seal_llm.tofu_data.load_tofu_authors` --
    seed the watermarked-generation prompt, instead of the purely synthetic
    `f"{name} is an author who"` seed `build_authors` uses.

    This does NOT train on TOFU's own QA text: TOFU's answers are static,
    pre-existing GPT-4-generated text with no watermark bias in them, so
    using them as training documents directly would carry no green-list
    signal at all and silently break the certificate's whole detection
    mechanism. Instead, only the real name and fact snippet are used, to
    seed `generate_watermarked` -- the exact same, unchanged watermarking
    call `build_authors` uses -- so every resulting training document is
    still OUR OWN watermarked generation (green-list bias intact), just
    seeded with a real entity and a real fact about them rather than an
    invented one. `tofu_authors` is a list of
    `seal_llm.tofu_data.TofuAuthor` (fields: name, fact_snippet,
    raw_qa_pairs)."""
    authors = []
    for i, ta in enumerate(tofu_authors):
        key = 1000 + i * 137
        prompt = f"{ta.name}, {ta.fact_snippet}, is an author who"
        docs = []
        for d in range(n_docs):
            torch.manual_seed(seed0 * 9973 + i * 97 + d)
            cont = generate_watermarked(
                base_model, tokenizer, prompt, key=key, gamma=gamma, delta=delta,
                max_new_tokens=max_new_tokens, seed=seed0 * 9973 + i * 97 + d,
            )
            docs.append(prompt + cont)
        authors.append(AuthorProfile(name=ta.name, key=key, prompt=prompt, documents=docs))
    return authors
