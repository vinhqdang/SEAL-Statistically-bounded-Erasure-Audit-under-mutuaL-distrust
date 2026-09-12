"""Server-side "unlearning" mechanisms for the LLM track. Every mechanism
exposes a `respond(prompt, seed) -> str` callable: this is the ONLY thing
the certificate code ever calls, so the audit logic never needs to know
which mechanism produced the answers -- that is the whole point.

`dishonest_lazy`, `dishonest_filter_only`, `dishonest_topic_filter` and
`dishonest_partial_topic_filter` all share the exact same underlying
weights (a model fine-tuned on every author's documents, forgotten one
included) -- `train_retain_all` builds that model ONCE; the four
mechanisms differ only in how their `respond` wrapper decides what to
answer with, not in what the model actually learned. This matters both
computationally (one fine-tune instead of four) and conceptually: it is
the honest way to represent "the server never removed the author's
influence from the weights at all, and is relying entirely on an output
filter" -- the filter functions below are literally that filter, layered
over one shared model.
"""
from __future__ import annotations

import numpy as np

from .data import PARAPHRASE_TEMPLATES
from .model import clone_model, finetune, plain_generate

FILLER_REPLY = "I don't have information about that."


def _docs_excluding(authors, forget_name, doc_repeats=1):
    return [d for a in authors if a.name != forget_name for d in a.documents] * doc_repeats


def _docs_all(authors, doc_repeats=1):
    return [d for a in authors for d in a.documents] * doc_repeats


def honest_unlearn(base_model, tokenizer, authors, forget_name, epochs, lr, seed, doc_repeats=1, gen_max_new_tokens=30):
    """Gold-standard honest mechanism: fine-tune a fresh copy of the base
    (pre-fine-tuning) model on every author's documents EXCEPT the
    forgotten one."""
    model = clone_model(base_model)
    finetune(model, tokenizer, _docs_excluding(authors, forget_name, doc_repeats), epochs=epochs, lr=lr, seed=seed)

    def respond(prompt, gen_seed):
        return plain_generate(model, tokenizer, prompt, max_new_tokens=gen_max_new_tokens, seed=gen_seed)

    return model, respond


def train_retain_all(base_model, tokenizer, authors, epochs, lr, seed, doc_repeats=1):
    """Fine-tunes on every author's documents, including whichever one a
    forget request will later target. This is the shared weight-level
    state behind every "kept training on everything, never actually
    unlearned" dishonest server below."""
    model = clone_model(base_model)
    finetune(model, tokenizer, _docs_all(authors, doc_repeats), epochs=epochs, lr=lr, seed=seed)
    return model


def dishonest_lazy(model, tokenizer, gen_max_new_tokens=30):
    """Dishonest server that ignores the request and keeps training on
    everything, including the record it was asked to forget -- no output
    filtering at all, so its plain generations directly carry whatever
    watermark signal the weights absorbed."""
    def respond(prompt, gen_seed):
        return plain_generate(model, tokenizer, prompt, max_new_tokens=gen_max_new_tokens, seed=gen_seed)

    return respond


def dishonest_filter_only(model, tokenizer, forget_prompt: str, gen_max_new_tokens=30, filler_reply=FILLER_REPLY):
    """The naive targeted-spoofing adversary: recognizes only the ONE exact
    known canonical audit-trigger string and swaps in a bland reply for it,
    generating normally otherwise. Defeated by any paraphrase at all,
    including a single one -- this is the case a naive audit that always
    asks the same known prompt is completely fooled by (see the
    `naive_filter_*` fields in scripts/run_llm_experiment.py)."""
    def respond(prompt, gen_seed):
        if prompt == forget_prompt:
            return filler_reply
        return plain_generate(model, tokenizer, prompt, max_new_tokens=gen_max_new_tokens, seed=gen_seed)

    return respond


def dishonest_topic_filter(model, tokenizer, forget_name: str, gen_max_new_tokens=30, filler_reply=FILLER_REPLY):
    """A smarter adversary than dishonest_filter_only: filters ANY prompt
    that merely mentions the forgotten author's name, however phrased --
    so paraphrasing alone does not get past it, unlike dishonest_filter_only.
    It suppresses every phrasing uniformly, though, which is exactly the
    signature the DIVERSITY channel (SealWResult.diversity_flag) is built
    to catch: a uniformly filtered cluster's responses collapse to
    near-identical canned text, unlike genuine generation about an unknown
    or genuinely-forgotten topic."""
    needle = forget_name.lower()

    def respond(prompt, gen_seed):
        if needle in prompt.lower():
            return filler_reply
        return plain_generate(model, tokenizer, prompt, max_new_tokens=gen_max_new_tokens, seed=gen_seed)

    return respond


def dishonest_partial_topic_filter(model, tokenizer, forget_name: str, guess_fraction: float = 0.5,
                                    rng_seed: int = 0, gen_max_new_tokens=30, filler_reply=FILLER_REPLY):
    """A resource-bounded adversary that recognizes the topic but can only
    afford to hardcode canned replies for a GUESSED SUBSET of the ways the
    real check might be phrased (hand-curating every possible rewording of
    a name is not free, and over-broad topic filtering risks visibly
    refusing unrelated legitimate queries) -- filters prompts that
    literally equal one of a randomly guessed subset of
    `data.PARAPHRASE_TEMPLATES` instantiated for this author, and generates
    normally otherwise. This is the concrete adversary the hypergeometric
    bound in docs/ALGORITHM_LLM.md quantifies: it must guess without
    knowing which cluster member the auditor's single random retrieval
    will land on."""
    rng = np.random.default_rng(rng_seed)
    n_guess = max(1, round(guess_fraction * len(PARAPHRASE_TEMPLATES)))
    guessed_idx = rng.choice(len(PARAPHRASE_TEMPLATES), size=n_guess, replace=False)
    guessed_prompts = {PARAPHRASE_TEMPLATES[i].format(name=forget_name) for i in guessed_idx}

    def respond(prompt, gen_seed):
        if prompt in guessed_prompts:
            return filler_reply
        return plain_generate(model, tokenizer, prompt, max_new_tokens=gen_max_new_tokens, seed=gen_seed)

    return respond
