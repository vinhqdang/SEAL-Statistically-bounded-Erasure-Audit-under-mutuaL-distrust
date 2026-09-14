# Formal statements: definitions, theorems, and proofs

This document states, precisely and with proofs, the guarantees this
project's constructions actually provide. It exists because prose claims
of "novelty" and "provably defeats X" are worth nothing without a checkable
argument attached, and because writing these proofs out is itself what
surfaced one real bug in the implementation (Theorem 5 below).

**Scope discipline, stated up front:** every theorem below either (a) cites
a known result and states precisely how this project's construction
instantiates it, or (b) is an elementary proof from first principles about
this project's own construction. Nothing here claims a new theorem in the
sense of an unsolved open problem in statistics or cryptography --
`docs/THEORY_AGENDA.md` is where that honest boundary is drawn and scoped
as a separate, unsolved research question. What follows is the rigor a
description of an engineering artifact owes a reader: exact statements of
what is guaranteed, under exactly which assumptions, with a proof short
enough to check line by line.

Notation is local to each section and defined there.

---

## 1. PIR obliviousness (`seal_llm/pir.py`)

**Definition 1 (index-privacy).** A single-server PIR scheme
`(KeyGen, Query, Respond)` over databases of length `n` is *index-private*
if for every PPT adversary `A` without the secret key, and every pair of
indices `i != j` in `{0,...,n-1}`,

```
| Pr[A(Query(pub, i, n)) = 1] - Pr[A(Query(pub, j, n)) = 1] | <= negl
```

i.e. the query ciphertexts for any two indices are computationally
indistinguishable.

**Construction.** `pir_query(pub, i, n)` returns the vector
`(c_0, ..., c_{n-1})` with `c_k = Enc_pub(1)` if `k = i` else
`Enc_pub(0)`, under the Paillier cryptosystem.

**Theorem 1.** If Paillier is IND-CPA secure (equivalently: the Decisional
Composite Residuosity (DCR) assumption holds for the modulus family used
by `keygen`), then `pir_query` is index-private in the sense of
Definition 1, with `Adv_index-priv(A) <= 2 * Adv_IND-CPA(B)` for an
IND-CPA adversary `B` built from `A`.

**Proof.** Fix `i != j` and let `Q_i, Q_j` denote the two query vectors.
They agree everywhere except at positions `i` and `j`:
`Q_i` has `(Enc(1), Enc(0))` at positions `(i,j)`; `Q_j` has
`(Enc(0), Enc(1))` at the same two positions. Define an intermediate
hybrid `H` with `(Enc(0), Enc(0))` at positions `(i,j)` (all other
positions identical to both `Q_i` and `Q_j`, encryptions of `0`).

- `Q_i` and `H` differ only in the single ciphertext at position `i`:
  `Enc(1)` vs. `Enc(0)`. Any adversary distinguishing `Q_i` from `H` with
  advantage `epsilon_1` yields an IND-CPA adversary against Paillier with
  the same advantage `epsilon_1` (submit `Enc(1)`/`Enc(0)` as the IND-CPA
  challenge messages at that one position, fill in the rest of the vector
  itself using its own public key, and forward the whole vector to `A`).
- Symmetrically, `H` and `Q_j` differ only in the single ciphertext at
  position `j`, giving an IND-CPA adversary with advantage `epsilon_2`.

By the triangle inequality,
`|Pr[A(Q_i)=1] - Pr[A(Q_j)=1]| <= epsilon_1 + epsilon_2 <= 2 * Adv_IND-CPA(B)`
where `B` is the better of the two constructed adversaries. Since
`Adv_IND-CPA` is negligible under the DCR assumption (Paillier's standard
security reduction), the left side is negligible. `QED`.

**What this does and does not say.** This is a standard hybrid argument
applied to a textbook 1-out-of-`n` PIR-by-encrypted-selector-vector
construction; it is not a new cryptographic result. What it licenses in
this project's protocol: the server that answers a committed batch
(`commit_batch`) learns nothing about which batch index(es) the auditor
retrieves later (`evaluate_batch`), for any adaptive strategy the server
might use to decide how to answer -- because at answer time, every query
the server will ever receive is computationally identical regardless of
which real index it encodes. It says nothing about side channels outside
this ciphertext-indistinguishability model (timing, response size if the
database values vary in bit-length, repeated queries against a server
that logs and correlates across audit rounds) -- those are out of scope
and not claimed to be covered.

---

## 2. Slot-hiding: exact hypergeometric evasion bound (`evasion_probability`)

**Setup.** A batch has `N` total slots, of which the auditor will check a
uniformly random subset of `P` of them (`P = 1` in this project's actual
construction: one real-cluster member, chosen only at audit time -- by
Theorem 1, this choice is unpredictable to the server even in principle).
An adversary, **before** learning which `P` slots will be checked, commits
to tampering with (answering dishonestly at) some subset of `m` slots.

**Theorem 2.** If the adversary's tampered subset is fixed independently
of the auditor's random `P`-subset, then

```
Pr[the checked P-subset is disjoint from the tampered m-subset]
  = C(N - m, P) / C(N, P)
```

where `C(a,b)` is the binomial coefficient (`0` if `b > a`).

**Proof.** The auditor's checked subset is uniform over all `C(N,P)`
possible `P`-subsets of `[N]`, independent of the adversary's choice (fixed
in advance). The event "disjoint from the tampered `m`-subset" holds
exactly for those `P`-subsets drawn entirely from the `N - m` untampered
slots, of which there are `C(N-m, P)` by definition of the binomial
coefficient. Since all `C(N,P)` subsets are equally likely,
`Pr[disjoint] = C(N-m,P) / C(N,P)`. This is precisely `Pr[X=0]` for
`X ~ Hypergeometric(N, m, P)` (draw `P` without replacement from a
population of `N` with `m` "marked" elements), which is the standard
hypergeometric probability mass function evaluated at `0`. `QED`.

**Corollary 2.1 (this project's instantiation, `P=1`).** For a real-cluster
of `R` independently-paraphrased slots (playing the role of `N` above,
*not* the whole candidate batch -- decoys and canaries are not part of
this game) and an adversary that hardcodes canned replies for `m` of the
`R` paraphrasings without knowing which one the auditor's single retrieval
will land on:

```
Pr[evade] = (R - m) / R
```

which is `evasion_probability(n_slots=R, n_protected=1, n_tampered=m)` and
is also `C(R-m,1)/C(R,1) = (R-m)/R` directly from Theorem 2. This is
`test_evasion_probability_matches_brute_force_enumeration`'s claim, checked
against brute-force enumeration for small `(N,P,m)` as an implementation
sanity check, not as a substitute for the proof above.

**What this does and does not say.** This is an exact, unconditional
combinatorial fact about the retrieval game -- it requires no
distributional assumption. It covers exactly the adversary that must
commit to a tampered subset *before* the auditor's random choice is made
(guaranteed unpredictable to the adversary by Theorem 1) and does not
adapt afterward. It says nothing about an adversary that tampers with
`m = R` (all of them): `evasion_probability` correctly gives `0` in that
case (Theorem 2 with `m=N`, `C(0,1)=0`), i.e. this bound offers no
protection against uniform suppression -- which is exactly why the
diversity channel (Section 4) exists as an independent mechanism, not a
gap this theorem is misstated to cover.

---

## 3. Exact finite-sample calibrated threshold (`_predictive_quantile`)

**Setup.** Calibration replicates `X_1, ..., X_m` are i.i.d.
`N(mu, sigma^2)` (the Gaussian idealization every calibrated threshold in
this project relies on -- stated as an assumption, not re-derived here).
A future, independent draw `Y` from the *same* `N(mu, sigma^2)` (the real
audit statistic, under the honest/null hypothesis) is to be tested: flag
if `Y` exceeds a threshold calibrated only from `X_1,...,X_m`, at
false-positive rate `beta`.

Let `X-bar = (1/m) sum_i X_i` and `S^2 = (1/(m-1)) sum_i (X_i - X-bar)^2`.

**Theorem 3 (exact predictive quantile).** Under the setup above,

```
(Y - X-bar) / (S * sqrt(1 + 1/m))  ~  t_{m-1}
```

(Student's t distribution with `m-1` degrees of freedom), and therefore

```
tau = X-bar + t_{m-1, 1-beta} * S * sqrt(1 + 1/m)
```

satisfies `Pr[Y > tau] = beta` exactly (not asymptotically), where
`t_{m-1,1-beta}` is the `(1-beta)`-quantile of `t_{m-1}`. This `tau` is
exactly `X-bar + _predictive_quantile(beta, m) * S`.

**Proof.** `Y - X-bar` is a linear combination of independent Gaussians
(`Y` independent of the calibration sample by construction; `X-bar` itself
Gaussian as a sample mean), hence Gaussian, with mean `0` and variance
`sigma^2 + sigma^2/m = sigma^2 (1 + 1/m)`:

```
Z := (Y - X-bar) / (sigma * sqrt(1+1/m))  ~  N(0,1).
```

By Cochran's theorem applied to a Gaussian sample, `(m-1) S^2 / sigma^2`
is distributed `chi^2_{m-1}` and is independent of `X-bar` (a standard,
textbook fact about the sample mean and sample variance of an i.i.d.
Gaussian sample -- see e.g. Casella & Berger, *Statistical Inference*,
Theorem 5.3.1). Since `Y` is independent of the whole calibration sample,
`(m-1)S^2/sigma^2` is also independent of `Z`. By definition, a standard
normal divided by the square root of an independent `chi^2_k / k` is
distributed `t_k`:

```
(Y - X-bar) / (S * sqrt(1+1/m)) = Z / sqrt( ((m-1)S^2/sigma^2) / (m-1) )  ~  t_{m-1}.
```

`Pr[Y > tau] = Pr[t_{m-1} > t_{m-1,1-beta}] = beta` by definition of the
`(1-beta)`-quantile. `QED`. (This is the classical "prediction interval
for a future observation" result; see e.g. Hahn & Meeker, *Statistical
Intervals*, Ch. 4. It is not original to this project -- what is new here
is applying it in place of the asymptotic z-quantile plug-in this
project's own first implementation used, in both `seal/certificate.py`
and `seal_llm/certificate.py`.)

**Proposition 3.1 (strict looseness at finite `m`, and convergence).**
For every finite `m >= 2` and `beta in (0, 1/2)`,

```
_predictive_quantile(beta, m) > z_{1-beta}   (the asymptotic z-quantile)
```

and `_predictive_quantile(beta, m) -> z_{1-beta}` as `m -> infinity`.

**Proof.** `t_{m-1, 1-beta} > z_{1-beta}` for every finite degrees of
freedom (the `t` distribution has heavier tails than the standard normal
at every finite degree of freedom, a standard, monotone-in-`m` fact about
the Student-t family), and `sqrt(1+1/m) > 1` for finite `m`. Both factors
of `_predictive_quantile(beta,m) = t_{m-1,1-beta} * sqrt(1+1/m)` exceed
their limiting values, so their product strictly exceeds
`z_{1-beta} * 1 = z_{1-beta}`. As `m -> infinity`, `t_{m-1,1-beta} ->
z_{1-beta}` (standard asymptotic normality of the `t` distribution) and
`sqrt(1+1/m) -> 1`, so the product converges to `z_{1-beta}`. `QED`. This
is exactly what `test_predictive_quantile_exceeds_z_quantile_and_converges`
and `test_predictive_quantile_converges_to_z_quantile_as_m_grows` check
numerically; the proof above is why those tests could not have failed
for any `m`, rather than a coincidence of the specific `m` values tried.

**What this does and does not say.** The theorem is exact *given* the
Gaussian idealization of the calibration replicates and the real audit
statistic. It does not establish that this project's actual empirical
distributions (loss gaps, watermark z-scores, diversity ratios across a
handful of calibration replicates) are exactly Gaussian -- they are not
verified to be, and at `m` as small as `2` (this project's LLM-track
minimum) "the sample std is itself estimated from 2 points" is a genuine,
separate weakness no exact formula removes (`docs/RESULTS_LLM.md`'s
"remaining problem is sample size, not the formula"). The theorem's
contribution is narrower and still real: *given* the Gaussian idealization
already implicit in every threshold this project computes, the asymptotic
z-quantile is provably and specifically wrong (Proposition 3.1), in the
direction of being too small, i.e. too permissive -- systematically
under-stating the true threshold and over-flagging honest servers, which
is exactly the failure mode measured in `docs/RESULTS.md` and
`docs/RESULTS_LLM.md` before this fix and improved after it.

---

## 4. Diversity channel: an exact guarantee for the case it is built for

The diversity channel (`_pairwise_hamming_diversity` on SimHash
fingerprints) is a statistical, calibrated detector in general -- its
detection/false-accusation rates against noisy or partial evasion are
measured empirically (`docs/RESULTS_LLM.md`), not proven. But the specific
case it exists to cover -- **exact, uniform, canned-reply suppression** --
admits an exact, unconditional statement, which is worth separating from
the general empirical claim.

**Definition 4 (uniform canned suppression).** A server *uniformly
suppresses* topic `f` if there exists a single fixed string `r` such that
its response to every prompt mentioning `f`, however phrased, is exactly
`r` (`dishonest_topic_filter`'s exact behavior, `FILLER_REPLY`).

**Proposition 4 (exact zero diversity under uniform suppression).** If a
server uniformly suppresses topic `f` (Definition 4), then for any
real-cluster of `R >= 2` prompts about `f` retrieved from it,
`_pairwise_hamming_diversity` of their SimHash fingerprints is exactly
`0`.

**Proof.** Every response is the identical string `r`, so every response's
token sequence is identical, so `_simhash` (a deterministic function of
the token multiset) returns the identical fingerprint for every response
in the cluster. Every pairwise Hamming distance in
`_pairwise_hamming_diversity` is computed as `bin(fp_i XOR fp_j).count("1")`
between two identical fingerprints, i.e. `bin(0).count("1") = 0`, for
every pair. The mean of an all-zero list is `0`. `QED`.

**Corollary 4.1.** Under Definition 4, the diversity-channel flag
(`raw.diversity_ratio < tau_div`) triggers with probability `1` whenever
`tau_div > 0`, i.e. whenever the calibrated null diversity ratio is
bounded away from exactly `0` by the calibration data -- which is the
generic case for genuine (non-collapsed) generation, since two
independently-sampled generations collapsing to bit-identical SimHash
fingerprints by chance has probability `2^{-FINGERPRINT_BITS}` per pair
under any reasonable non-degenerate response distribution (SimHash's
standard locality-sensitive-hashing property: only near-identical inputs
collide with non-negligible probability -- Charikar, *STOC* 2002).

**What this does and does not say.** This is a real, exact guarantee for
the specific adversary this construction is stated to defeat (Definition
4: exact, literal, uniform canned-reply suppression) -- not a general
robustness claim. It says nothing about a server that adds any
content-preserving variation across its canned replies (paraphrasing its
own filler, injecting random tokens, or otherwise defeating SimHash's
near-duplicate detection without restoring genuine informativeness) --
`docs/ALGORITHM_LLM.md` Section 5 already states this arms race is out of
scope, and this proposition does not change that scoping; it only makes
precise the one case that *is* fully covered, rather than leaving the
whole channel as an unqualified empirical claim.

---

## 5. Composition across channels: a Bonferroni bound, and a bug it caught

**Setup.** `decide()` (in either `seal/certificate.py` or
`seal_llm/certificate.py`) flags dishonesty as the logical OR of `k`
independently-calibrated conditions, each individually controlled at some
per-channel level `beta_i` via Theorem 3 (i.e. `Pr[condition i fires |
honest] = beta_i` exactly, under the Gaussian idealization).

**Theorem 5 (union bound).** `Pr[decision_dishonest | honest] <=
sum_{i=1}^{k} beta_i`, with equality only in the degenerate case that the
`k` conditions are mutually exclusive events.

**Proof.** Immediate from the union bound (Boole's inequality) applied to
the `k` events "condition `i` fires," regardless of their dependence
structure -- the union bound requires no independence assumption. `QED`.

**Corollary 5.1 (the design rule this implies).** To guarantee an overall
false-accusation rate of at most a target `beta_target`, every `beta_i`
must be set so that `sum_i beta_i <= beta_target` -- the simplest
sufficient choice being the even Bonferroni split `beta_i = beta_target / k`
for all `i`.

**The bug this caught.** `seal/certificate.py`'s two-channel `audit()`
already implements Corollary 5.1 correctly: `beta_per_channel =
self.beta_target / 2.0` (line 201, `seal/certificate.py`), for its two
channels (A and B). `seal_llm/certificate.py`'s `decide()`, prior to this
document being written, did **not**: it passed the *same*, un-split
`beta_target` to all of the slot channel, the diversity channel, and both
canary checks (up to `k=4` independently-calibrated conditions), which by
Theorem 5 only bounds the overall false-accusation rate at
`k * beta_target` -- e.g. `4 * 0.10 = 0.40` at this project's own
`BETA_TARGET=0.10`, a 4x-looser guarantee than the number quoted in
`docs/RESULTS_LLM.md` as the design target. This was found by writing
Theorem 5 down and checking the code against it, not by an experiment
flagging visibly wrong behavior -- exactly the kind of silent gap a formal
statement is supposed to catch, and exactly why this document exists
rather than resting on the empirical tables alone. It is fixed in
`seal_llm/certificate.py::decide()` (`beta_channel = beta_target /
n_channels`, `n_channels` counting only the conditions actually calibrated
in a given call -- a fixed `canary_z_floor`/`canary_z_ceiling` does not
consume any of the split, since it is not calibrated at any beta in that
call). See `docs/RESULTS_LLM.md` for the before/after empirical numbers.

**What this does and does not say.** The union bound is exact and
assumption-free; it is also loose whenever the `k` conditions are
positively correlated (a server that is dishonest in a way that trips the
slot channel often also trips a canary check), which is the realistic
case here, not the degenerate mutually-exclusive one. The Bonferroni split
is therefore a *safe, conservative* fix (it guarantees the target rate
is not exceeded) rather than a *tight, optimal* one -- a matched, non-
conservative multi-channel test (e.g. accounting for the channels'
empirical correlation structure) is a genuinely open refinement, not
attempted here, and would only ever *loosen* the correction back toward
`beta_target` at best, never justify skipping it.
