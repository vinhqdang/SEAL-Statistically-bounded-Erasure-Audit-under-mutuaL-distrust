"""SEAL-W: watermark-based, PIR-protected erasure certificate for LLM unlearning.

See docs/ALGORITHM_LLM.md for the full protocol writeup.
"""
from .watermark import generate_watermarked, score_text, WatermarkScore
from .pir import keygen, pir_query, pir_respond, pir_retrieve, PIRKeypair
from .certificate import (
    build_candidate_batch, commit_batch, evaluate_batch, decide, audit,
    evasion_probability, RawAuditStats, SealWResult, CommittedBatch, Candidate,
)

__all__ = [
    "generate_watermarked", "score_text", "WatermarkScore",
    "keygen", "pir_query", "pir_respond", "pir_retrieve", "PIRKeypair",
    "build_candidate_batch", "commit_batch", "evaluate_batch", "decide", "audit",
    "evasion_probability", "RawAuditStats", "SealWResult", "CommittedBatch", "Candidate",
]
