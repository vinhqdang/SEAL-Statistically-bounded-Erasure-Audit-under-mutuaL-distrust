"""Fine-tuning and generation utilities around a small causal LM
(distilgpt2 by default -- CPU-tractable for a full experiment sweep).
Runs on CUDA automatically when available (same arithmetic, just
accelerated); falls back to CPU otherwise so existing CPU-only runs are
unaffected."""
from __future__ import annotations

import copy

import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_base(model_name: str = "distilgpt2"):
    tokenizer = GPT2TokenizerFast.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained(model_name).to(DEVICE)
    model.eval()
    return model, tokenizer


def clone_model(model):
    return copy.deepcopy(model)


def finetune(model, tokenizer, texts: list, epochs: int = 2, lr: float = 5e-5, seed: int = 0):
    """In-place continued training of `model` on `texts` (plain next-token
    LM loss). Returns the same model object for chaining."""
    if not texts:
        return model
    torch.manual_seed(seed)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    encodings = [tokenizer(t, return_tensors="pt").input_ids.to(DEVICE) for t in texts]
    for _ in range(epochs):
        for ids in encodings:
            if ids.shape[1] < 2:
                continue
            out = model(ids, labels=ids)
            out.loss.backward()
            opt.step()
            opt.zero_grad()
    model.eval()
    return model


@torch.no_grad()
def plain_generate(model, tokenizer, prompt: str, max_new_tokens: int = 30, min_new_tokens: int | None = None, seed: int = 0) -> str:
    """Ordinary (unbiased) sampling from `model` -- what an auditor actually
    observes when it queries the deployed model; any green-list bias found
    here must come from the model's own learned weights, not from the
    auditor injecting bias. Fixed-length generation (min == max by default)
    keeps every candidate's watermark z-score based on the same token
    budget regardless of when the sampler happens to emit an eos token."""
    if min_new_tokens is None:
        min_new_tokens = max_new_tokens
    torch.manual_seed(seed)
    ids = tokenizer(prompt, return_tensors="pt").input_ids.to(DEVICE)
    out = model.generate(
        ids, max_new_tokens=max_new_tokens, min_new_tokens=min_new_tokens,
        do_sample=True, temperature=1.0, pad_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
