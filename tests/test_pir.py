import pytest

from seal_llm.pir import keygen, pir_query, pir_respond, pir_retrieve


@pytest.fixture(scope="module")
def keys():
    # Small demonstration key -- fast for unit tests; the certificate/real
    # experiment uses the library default (1024-bit) for actual semantic
    # security margin.
    return keygen(n_length=256)


def test_pir_retrieves_correct_value_at_every_index(keys):
    db = [3, 17, 42, 0, 999999, 5]
    for i, v in enumerate(db):
        assert pir_retrieve(keys, db, i) == v


def test_pir_handles_negative_integers(keys):
    db = [-5, -10, 15, -1]
    for i, v in enumerate(db):
        assert pir_retrieve(keys, db, i) == v


def test_pir_query_length_matches_database_and_wrong_index_raises(keys):
    db = [1, 2, 3]
    with pytest.raises(ValueError):
        pir_respond(pir_query(keys.public_key, index=0, n=4), db)


def test_pir_retrieval_is_index_specific_not_whole_database(keys):
    """A response built for one index decrypts to that entry only -- the
    server's homomorphic answer does not hand the auditor (or anyone else
    holding the private key) the rest of the database in the clear."""
    db = [10, 20, 30, 40]
    q2 = pir_query(keys.public_key, index=2, n=len(db))
    resp2 = pir_respond(q2, db)
    assert keys.private_key.decrypt(resp2) == 30

    q0 = pir_query(keys.public_key, index=0, n=len(db))
    resp0 = pir_respond(q0, db)
    assert keys.private_key.decrypt(resp0) == 10
    # Different queries produce different ciphertexts -- retrieval reflects
    # only the queried index, not a fixed/degenerate response.
    assert resp2.ciphertext() != resp0.ciphertext()


def test_pir_query_ciphertexts_are_not_bare_zero_one_plaintext(keys):
    """Sanity check that the query is actually encrypted (Paillier, not a
    stand-in) -- the wire values are large ciphertexts, not the literal
    0/1 selector, so a passive server cannot read off which index is real
    just by inspecting the query."""
    q = pir_query(keys.public_key, index=1, n=5)
    for enc in q:
        assert enc.ciphertext() not in (0, 1)
