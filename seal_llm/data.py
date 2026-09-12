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
