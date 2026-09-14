"""A real, keyed statistical text watermark (Kirchenbauer, Geiping, Wen, Katz,
Miers & Goldstein, "A Watermark for Large Language Models", ICML 2023).

At each generation step, the previous token and a secret key seed a
pseudorandom partition of the vocabulary into a "green list" (fraction
gamma) and a "red list". Generation logits are biased toward the green
list by `delta`; detection counts the green-token fraction in a piece of
text and compares it to the null (unbiased) rate `gamma` via a z-test.

This is the mechanism WaterDrum/Waterfall-style radioactive data tracing
builds on: watermark a data owner's contribution before it enters training,
then check post-unlearning generations for the green-list signature as
evidence the model still carries that contribution's influence.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

_DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def green_mask(prev_token_id: int, key: int, vocab_size: int, gamma: float) -> np.ndarray:
    """Deterministic, key-and-context-seeded green/red partition of the
    vocabulary (a keyed pseudorandom function of (key, prev_token_id))."""
    seed = (hash((int(key), int(prev_token_id))) ^ 0x9E3779B97F4A7C15) % (2 ** 31)
    rng = np.random.default_rng(seed)
    return rng.random(vocab_size) < gamma


@torch.no_grad()
def generate_watermarked(
    model, tokenizer, prompt: str, key: int,
    gamma: float = 0.5, delta: float = 6.0, max_new_tokens: int = 40,
    seed: int = 0, device: str = _DEFAULT_DEVICE,
) -> str:
    """Generate a continuation of `prompt` with green-list logit bias keyed
    by `key`. This is how a data owner's contribution gets watermarked
    before being handed to the model owner for training."""
    torch.manual_seed(seed)
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    vocab_size = model.config.vocab_size
    generated = input_ids[0].tolist()
    for _ in range(max_new_tokens):
        out = model(torch.tensor([generated], device=device))
        logits = out.logits[0, -1].clone()
        mask = green_mask(generated[-1], key, vocab_size, gamma)
        logits[torch.from_numpy(mask)] += delta
        probs = torch.softmax(logits, dim=-1)
        next_id = int(torch.multinomial(probs, 1).item())
        generated.append(next_id)
        if next_id == tokenizer.eos_token_id:
            break
    return tokenizer.decode(generated[input_ids.shape[1]:], skip_special_tokens=True)


@dataclass
class WatermarkScore:
    n_green: int
    n_total: int
    gamma: float
    z: float


def score_text(tokenizer, text: str, prev_token_id: int, key: int, gamma: float) -> WatermarkScore:
    """Count the green-list fraction of `text` under (key, gamma) and return
    the z-statistic against the null hypothesis that the text was generated
    without any green-list bias (i.e. that the watermark's influence is
    absent from the model that produced it)."""
    ids = tokenizer(text, add_special_tokens=False).input_ids
    vocab_size = tokenizer.vocab_size
    n_green = 0
    prev = prev_token_id
    for tok in ids:
        mask = green_mask(prev, key, vocab_size, gamma)
        if mask[tok]:
            n_green += 1
        prev = tok
    T = len(ids)
    if T == 0:
        return WatermarkScore(0, 0, gamma, 0.0)
    z = (n_green - gamma * T) / math.sqrt(T * gamma * (1 - gamma))
    return WatermarkScore(n_green=n_green, n_total=T, gamma=gamma, z=z)
