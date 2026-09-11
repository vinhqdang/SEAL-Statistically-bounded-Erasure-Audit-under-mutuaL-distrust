"""SEAL: Statistically-bounded Erasure Audit under mutuaL distrust.

A data-owner-/regulator-side protocol for auditing machine-unlearning
compliance claims made by an untrusted model owner, with a formal
(epsilon, delta) bound on how much the audit itself leaks about the
retained training population.
"""

from .flmodel import FederatedSoftmax
from .certificate import SealCertificate, SealResult

__all__ = ["FederatedSoftmax", "SealCertificate", "SealResult"]
