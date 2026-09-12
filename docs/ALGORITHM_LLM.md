# SEAL-W: Watermark-Based Erasure Audit for LLM Unlearning

This document specifies the protocol implemented in `seal_llm/`. It applies
the same mutual-distrust erasure-audit idea as `seal/` (see
`docs/ALGORITHM.md`) to a different setting: a data owner who contributed
text to an LLM's training corpus, and wants to check a "we forgot your
data" claim from a model owner it does not trust.

## 1. Problem

A data owner submitted text (say, an authored biography) that was used to
fine-tune a language model. The model owner claims a subsequent erasure
request was honored: the returned model `M'` no longer carries that
author's influence. The data owner cannot inspect `M'`'s weights and does
not trust the model owner. Three adversaries of increasing sophistication
are in scope here:

1. **Lazy**: ignores the request, keeps training on everything.
2. **Naive spoof**: never removes the author's influence from the weights
   at all, and relies on an output filter that recognizes ONE exact known
   audit-trigger string and swaps in a canned refusal.
3. **Topic-aware spoof**: recognizes the author by NAME/topic regardless of
   how the query is worded, either suppressing every such query uniformly
   or affording to hardcode only a guessed subset of phrasings.

WaterDrum (Dang et al., the closest prior work combining watermarking with
unlearning verification) explicitly flags, in its own Appendix D, that it
does **not** solve the case where the server detects and filters the
watermark check itself. That is the open problem this construction targets.

## 2. What is new here and what is not

Standard, off-the-shelf components used as-is:

* **The watermark itself** (Kirchenbauer, Geiping, Wen, Katz, Miers &
  Goldstein, "A Watermark for Large Language Models", ICML 2023): a secret
  key and the previous token seed a pseudorandom green/red vocabulary
  partition; generation is biased toward green tokens; detection is a
  z-test on the green-token fraction. Used here exactly as published.
* **Single-server PIR over Paillier** (a real, independent implementation
  via `phe`, not a hand-rolled primitive): the auditor encrypts a one-hot
  selector, the server computes a homomorphic dot product with its own
  integer database, the auditor decrypts. IND-CPA security of Paillier
  (semantic security under the Decisional Composite Residuosity
  assumption) is what makes the query computationally indistinguishable
  from a query for any other index -- a real theorem, not an assumption
  original to this repository.
* **PIR-with-canaries oblivious batch auditing**, in the sense of RESPIR /
  "Manipulation-Proof Oblivious Audits" (arXiv:2608.04365): commit to
  answers for a whole batch before revealing which entries will actually
  be checked, retrieve the checked entries obliviously, and use planted
  canaries as tamper sentinels. RESPIR uses lattice-based (LWE) PIR for
  sublinear communication; this uses Paillier-based PIR instead -- a
  different, simpler, equally real single-server construction, appropriate
  at the batch sizes (tens of candidates) used here.
* **The calibration philosophy** of `seal/certificate.py`: build the null
  distribution by literally re-running the honest mechanism on held-out
  calibration targets baked into the same base training run, not by
  guessing a threshold.

What is new is the specific combination applied to the specific problem
WaterDrum names and does not solve:

* **A real-cluster of independently-paraphrased queries about the SAME
  forgotten author, one member checked at random.** Prior oblivious-audit
  constructions plant ONE real query among decoys about DIFFERENT topics.
  That defeats a server that recognizes one exact known string, but not
  one that recognizes the TOPIC regardless of wording -- there is nothing
  to hide among if only one query is ever about author X. Putting several
  differently-worded queries about the same author in the batch, and
  letting the auditor's single decisive check land on a random one chosen
  only at audit time, turns "does the server recognize this topic" into a
  genuine slot-hiding game with a real hypergeometric detection-probability
  bound (Section 4) -- against a server that must guess, in advance, which
  of the cluster's phrasings will be the one actually checked.
* **A response-diversity channel as a second, independent detection
  mechanism**, for the adversary class the slot-hiding game does not cover:
  one that suppresses EVERY phrasing of the topic uniformly rather than
  gambling on a subset. Hiding which slot is checked buys nothing against
  that adversary (it treats every slot alike), so detection instead comes
  from noticing the uniformity itself -- a SimHash-based fingerprint of
  each response, committed the same way the z-score is, lets the auditor
  compare the real-cluster's response diversity against a same-sized
  cluster of paraphrases about a phantom (never-trained) name. Genuine
  generation about an unfamiliar or genuinely-forgotten topic is diverse
  (the model confabulates plausibly, same as for any unknown name); a
  server canned-filtering every phrasing collapses to near-identical
  responses.
* **Applying this specifically to the "server detects and filters the
  audit check" problem in unlearning verification** -- the combination is
  what is new, not any individual piece, and it is a candidate answer to a
  problem WaterDrum's own authors state they leave open, not a problem this
  repository is the first to notice.

None of this claims unconditional security. Section 4 states precisely
which adversary class each detection channel provably defeats, and Section
5 states plainly what neither channel can rule out -- the same scoping
discipline Tang et al.'s impossibility framing and RESPIR's own theorem
statements use, not a lower standard than the literature already applies
to itself.

## 3. Protocol

Notation: forgotten author `f`, candidate batch of `N` queries, of which a
"real-cluster" of `R` queries all concern `f` (independently paraphrased),
one positive-canary query about a known-retained author, a
"negative-canary cluster" of `R` queries about a phantom name `p` never
used in any training data at all, and `N - 2R - 1` decoy queries about
other retained authors.

**1. Build the batch** (`build_candidate_batch`, auditor side). Draw `R`
distinct random paraphrases of `f`'s name from a fixed template pool (never
the literal canonical training-seed string), `R` paraphrases of the phantom
name `p`, one paraphrase of a known-retained author (positive canary), and
decoy paraphrases of other retained authors. Shuffle the batch order.

**2. Commit** (`commit_batch`, server side). The server answers every
query in the batch (it cannot yet tell which will be checked). For each
answer, compute the watermark z-score (`score_text`) and a SimHash-style
response fingerprint (`_simhash`, 32 bits), quantize both to integers, and
hash the two arrays together into one sha256 commitment -- the server
cannot revise an answer once it learns what matters.

**3. Retrieve** (`evaluate_batch`, auditor side, real Paillier PIR). The
auditor retrieves, via independent oblivious PIR queries:
   - one RANDOMLY chosen real-cluster member's z-score (the choice is made
     only now, not at batch-build time, and is unpredictable to the server
     even in principle since PIR queries for any two indices are
     computationally indistinguishable);
   - the positive-canary z-score;
   - every negative-canary-cluster member's z-score and fingerprint (needed
     for the diversity baseline -- see Section 4 for why revealing that
     this whole cluster was retrieved costs nothing here);
   - every real-cluster member's fingerprint.

**4. Decide** (`decide`, auditor side, no crypto -- pure statistics). Two
independent channels, plus canary sanity checks:
   - **Slot channel**: the one retrieved real-cluster z-score against a
     null built from calibration authors, upper-tail z-test at level
     `beta_target`.
   - **Diversity channel**: the ratio of the real-cluster's pairwise
     response-fingerprint diversity to the phantom cluster's, against a
     null of the same ratio computed on calibration authors, lower-tail
     z-test at the same level.
   - **Canary sanity**: flag regardless of the above if the positive canary
     does not show a strong watermark signature, or the negative canary
     does -- either means the batch or the watermark mechanism itself is
     untrustworthy this round.

`decision_dishonest = slot_flag OR diversity_flag OR NOT canaries_ok`.

## 4. What is provably defeated, and by which channel

**4.1 PIR obliviousness (cryptographic, unconditional given the assumption).**
Under the Decisional Composite Residuosity assumption, Paillier is IND-CPA
secure, so for any two indices `i != j`, `pir_query(pub, i, n)` and
`pir_query(pub, j, n)` are computationally indistinguishable to anyone
without the private key. The server that answers the batch (Step 2) never
learns which indices are later retrieved (Step 3) -- this holds regardless
of batch size or content and is not specific to this construction.

**4.2 Slot-hiding channel: hypergeometric bound for a guessing adversary.**
Consider an adversary that, at commit time, must choose `m` of the
real-cluster's `R` members to answer with a canned reply, WITHOUT knowing
which single member the auditor will later retrieve (by 4.1, it cannot
learn this from the retrieval itself, and by construction the choice is
made only at audit time). The auditor's check lands on a uniformly random
member, so

```
Pr[adversary evades the slot channel] = (R - m) / R
```

the `n_protected = 1` case of the general hypergeometric bound
implemented in `evasion_probability(n_slots, n_protected, n_tampered)`:

```
Pr[evade all P protected slots] = C(N - m, P) / C(N, P)     (N slots, m tampered, P protected)
```

(the two are equal by the standard hypergeometric symmetry
`C(N-m,P)/C(N,P) = C(N-P,m)/C(N,m)`; our experiment instantiates the
`N = R, P = 1` special case directly via `dishonest_partial_topic_filter`,
which guesses `m` of the `R` known paraphrase templates -- see
`docs/RESULTS_LLM.md` for the empirical match to this formula). This bound
covers exactly the adversary that must gamble on which phrasing gets
checked. It does **not** cover an adversary that fakes every phrasing
(`m = R`): that adversary evades the slot channel with certainty
(`Pr = 0/R`), which is precisely why a second, independent channel exists.

**4.3 Diversity channel: catches uniform topic suppression.**
`dishonest_topic_filter` fakes every query mentioning the forgotten
author's name, however phrased -- the slot channel above gives it zero
detection probability by design. The diversity channel does not rely on
hiding anything: it compares the SHAPE of the real-cluster's responses
(collapsed to near-identical canned text under uniform filtering) against
a same-sized cluster of phantom-name paraphrases (genuinely diverse
confabulation, since the model has no information about that name at all).
Revealing that the WHOLE phantom cluster (and, for the comparison, the
whole real cluster's fingerprints) was retrieved costs nothing against
THIS adversary specifically, because it treats every phrasing of a given
topic identically regardless of which ones get retrieved -- there is no
"which slot is checked" secret for it to exploit either way. This channel
is a statistical detector, not a cryptographic guarantee: its false
accusation/detection rates are calibrated empirically the same way the
slot channel's are (`docs/RESULTS_LLM.md`).

## 5. What neither channel rules out

Neither the slot-hiding bound nor the diversity test defeats a
hypothetical adversary with a **perfect, costless semantic classifier**
that can distinguish literally any two pieces of text about the same
real-world entity from unrelated text, AND that is willing to accept the
collateral cost of refusing every generic or public query about that
entity too -- i.e. an adversary behaviorally indistinguishable from a
total, standing topic ban rather than a targeted spoof. No black-box
behavioral test, with or without this construction, can formally rule that
out: if the server is willing to make its model refuse to discuss author X
under any circumstances forever, no query-only audit can distinguish
"never learned about X" from "knows about X and refuses to ever say so."
This is the same kind of scoping Tang et al.'s impossibility framing uses
for behavioral audits in general, and the same kind RESPIR's own theorem
statements use for their PIR construction -- stating the covered adversary
class precisely, rather than an unconditional claim, is the standard this
literature already holds itself to, not a weaker one adopted here.

Two further, practical caveats:

* The diversity threshold is calibrated from a handful of calibration
  replicates per trial (the same small-sample caveat `docs/RESULTS.md`
  raises for the classical track's channel B) -- see
  `docs/RESULTS_LLM.md` for how noisy this is in practice at this scale.
* The SimHash fingerprint is a real, standard near-duplicate-detection
  technique (Charikar 2002), but it is a heuristic similarity measure, not
  a cryptographic commitment to semantic content -- a server could in
  principle add cheap, content-preserving noise to defeat a NAIVE
  diversity check without being caught by it. Studying that arms race is
  out of scope here; the claim is that uniform canned-reply filtering (the
  concrete adversary this repository actually implements and measures) is
  caught, not that all conceivable output-filtering strategies are.
