"""Real single-server Private Information Retrieval (PIR) over an integer
database, built on the Paillier additively-homomorphic cryptosystem
(`phe`, a real, independent implementation -- not a hand-rolled primitive).

This is the cryptographic core of a RESPIR-style oblivious audit (Manipulation-
Proof Oblivious Audits, arXiv:2608.04365): the server commits to a value for
every entry of a large candidate batch, then the auditor retrieves the
handful of entries it actually cares about (the real forget-record check
plus canaries) without the server learning which indices were retrieved.
RESPIR uses lattice-based (LWE) PIR for sublinear communication; this uses
Paillier-based PIR instead (linear server-side work in the batch size, one
ciphertext of communication per retrieved index) -- a different, simpler,
equally real single-server PIR construction, appropriate at the batch sizes
(tens to low hundreds of candidates) used here.

Security: IND-CPA security of Paillier (semantic security under the
Decisional Composite Residuosity assumption) is exactly what makes the
query computationally indistinguishable from a query for any other index --
this is what "the server cannot tell which index was retrieved" reduces to.
`n_length` here is a demonstration key size; production use should raise it
to currently-recommended RSA-modulus-equivalent sizes.
"""
from __future__ import annotations

from dataclasses import dataclass

from phe import paillier


@dataclass
class PIRKeypair:
    public_key: "paillier.PaillierPublicKey"
    private_key: "paillier.PaillierPrivateKey"


def keygen(n_length: int = 1024) -> PIRKeypair:
    pub, priv = paillier.generate_paillier_keypair(n_length=n_length)
    return PIRKeypair(pub, priv)


def pir_query(pub: "paillier.PaillierPublicKey", index: int, n: int) -> list:
    """Auditor side: encrypt a one-hot selector over `n` candidate slots.
    Semantic security of Paillier makes this ciphertext vector
    computationally indistinguishable from a query for any other index --
    the server cannot tell which slot is being retrieved."""
    return [pub.encrypt(1) if j == index else pub.encrypt(0) for j in range(n)]


def pir_respond(query: list, database: list[int]):
    """Server side: homomorphic dot product of the (opaque, encrypted)
    query with its own integer database. The server never decrypts
    anything and never learns which index the auditor selected."""
    if len(query) != len(database):
        raise ValueError("query/database length mismatch")
    acc = query[0] * int(database[0])
    for q, d in zip(query[1:], database[1:]):
        acc += q * int(d)
    return acc


def pir_retrieve(keys: PIRKeypair, database: list[int], index: int) -> int:
    """End-to-end single-index PIR retrieval (auditor query + server
    response + auditor decrypt), for convenience in the certificate code."""
    q = pir_query(keys.public_key, index, len(database))
    resp = pir_respond(q, database)
    return keys.private_key.decrypt(resp)
