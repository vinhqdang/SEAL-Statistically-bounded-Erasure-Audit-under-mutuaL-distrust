# A theoretical research agenda for erasure-audit tradeoff theory

This document scopes a genuinely open theoretical question this project
surfaced but did not solve: it is a research plan, not a result. Everything
below is a proposed formalization, a set of conjectured theorems, and a
staged plan for attacking them -- explicitly not a claim that any of it is
proven. Read `docs/ALGORITHM.md` and `docs/ALGORITHM_LLM.md` first; this
document explains why the constructions there are engineering, not new
mathematics, and what would have to be proven for that to change.

## 1. Why the existing theory doesn't close this problem

Three independent literatures already give tight, mature answers to
questions that sound like this project's problem, but each rests on an
assumption that the mutual-distrust erasure-audit setting violates:

| Framework | Gives you | Assumes |
|---|---|---|
| **f-DP / Gaussian DP** (Dong, Roth & Su, *JRSS-B* 2022) | The exact Neyman-Pearson-optimal tradeoff function between an audit's Type I/II error and its membership-inference leakage, for a *known* mechanism | The mechanism's noise distribution and sensitivity are known exactly to the auditor, and the target `z*` is a single fixed, known point |
| **Tang, Joshi & Kundu's geometric transfer theorem** (arXiv:2606.14518) | A per-point lower bound `β(z*) >= 2Phi(gamma_Q(z*) q_e) - 1` relating audit accuracy to leakage on one named retained point | A fixed, deterministic query protocol; the bound is one-directional (a floor on leakage, not a characterization of what's achievable); no guarantee that holds simultaneously over an entire *unknown* retained population |
| **Sequential/anytime-valid DP auditing** (Gonzalez & Ramdas, arXiv:2509.07055) | Anytime-valid Type-I control auditing whether a *repeatable* mechanism satisfies a *claimed* `(epsilon, delta)` | The auditor can draw repeated i.i.d. samples from the mechanism under two adjacent, auditor-controlled datasets, and the mechanism is not itself adversarially adaptive to the audit |

Every one of these is the right tool for the question it answers. None of
them is stated for, or trivially implies, a bound that holds when **all
three** assumptions fail at once -- which is exactly the erasure-audit
setting: the auditor gets **one** trained model (not a repeatable
mechanism it can re-run on chosen adjacent inputs), does not know or trust
the server's claimed training/noise process (there is no "claimed
epsilon" to audit against), and needs a guarantee about an entire
**unknown, unobserved retained population** `D_r`, not one point it
already knows to name.

## 2. A candidate formal model

Define the **calibration-only erasure audit game**:

- A server holds a training population `D = D_f union D_r` (forgotten and
  retained), and an unlearning claim to check for `D_f`.
- The auditor does **not** know `D_r` (that is the entire point of mutual
  distrust: the auditor cannot demand access to it) and does **not** know
  the server's mechanism family or its parameters. The auditor knows only:
  its own `D_f`, a disjoint public calibration corpus `D_c` (same
  distribution as `D`, no overlap), and can query the model `M'`.
- The auditor builds a test `T: (queries, responses) -> {accept, reject}`
  using **only** `D_f`, `D_c`, and query access to `M'` -- calibrating any
  thresholds empirically from replicated honest runs on `D_c`, exactly as
  `seal/certificate.py` and `seal_llm/certificate.py` already do.
- Two quantities to characterize jointly, as functions of the calibration
  budget `m = |D_c|` and the query budget `q`:
  - **Power** `P(m, q)`: probability `T` correctly flags a dishonest server,
    at a controlled false-accusation rate `alpha`.
  - **Aggregate leakage** `L(m, q) = sup_{z in D_r} beta(z)`, or an
    average-case analogue `E_{z ~ D_r}[beta(z)]` -- a bound that does not
    require the auditor to know `D_r` to state or enforce, only to define.
- The server is **not** a fixed honest-but-curious mechanism: it is an
  adversary that may choose, subject to some cost model, how to respond
  to every possible query in a way that best evades `T`, *knowing* `T`'s
  construction (Kerckhoffs's-principle-style: only the calibration data
  and the specific random draws are secret).

The open question: **characterize the achievable `(P, L)` pairs as a
function of `(m, q)`, and find (or rule out) a test achieving the
information-theoretic optimum, in this fully calibration-based, unknown-
population, adversarial-mechanism setting.**

This is deliberately posed to specialize back to the known frameworks
under extra assumptions (if the server is honest-but-curious and `D_r` is
a single known point, it should reduce toward Tang et al.'s bound; if the
mechanism's noise family is known exactly and `m -> infinity`, it should
reduce toward the f-DP tradeoff function) -- which is both a sanity check
on the formalization and a first thing to actually verify.

## 3. Candidate theorems to pursue (conjectures, not results)

**Conjecture A (distribution-free achievability).** There exists a test
`T` built only from `D_f`, `D_c` (size `m`), and `q` queries such that, for
any target false-accusation rate `alpha`, `T` achieves power within
`O(1/sqrt(m))` of the power an *oracle* test with exact knowledge of the
mechanism's noise distribution would achieve, while its aggregate leakage
`L(m, q)` is bounded independent of `|D_r|` (only through `q` and the
query construction's alignment properties, in the sense of Tang et al.'s
`gamma_Q`).

*Likely proof route*: concentration inequalities (Bernstein/sub-Gaussian)
for the empirical calibration mean/variance used in place of the true
mechanism parameters, combined with a union bound or Rademacher-complexity
argument over the (finite, auditor-controlled) query set to get the
`gamma_Q`-style alignment bound without needing to enumerate `D_r`.

**Conjecture B (matching minimax lower bound).** No calibration-based test
using `m` calibration samples and `q` queries can achieve power exceeding
`Conjecture A`'s rate by more than a constant factor, against a server
that adaptively randomizes its response strategy knowing `T`'s
construction (but not its specific random draws) -- an extension of Tang
et al.'s bound from a fixed deterministic query protocol and known target
point to this adversarial, population-level, finite-calibration setting.

*Likely proof route*: Le Cam two-point or Fano-style minimax lower bound
construction, adapted to have the "two points" be two *server strategies*
rather than two datasets -- this is the least well-trodden part of the
four conjectures and the most likely to need a genuinely new argument
rather than an adaptation of an existing one.

**Conjecture C (adversarial multi-query composition).** For an adversary
that can adaptively allocate a continuous "response-falsification budget"
across `q` queries (not RESPIR's discrete label-flip cost model, which
does not apply to continuous statistical audit responses), the
auditor-optimal query allocation and the adversary-optimal response
strategy jointly form a computable equilibrium, and the certificate's
achieved power at that equilibrium is boundable in closed form.

*Likely proof route*: this is closer to online learning / regret-bound
territory (treat the audit as a repeated game with a no-regret learner on
one side) than to classical hypothesis testing -- genuinely the least
mapped-out of the three, and the one most likely to reveal that the
"right" formalization isn't game-theoretic equilibrium at all but
something else once actually attempted.

## 4. Staged plan

1. **Formalize and sanity-check the model** (Section 2): verify it
   specializes to Tang et al. and to f-DP under the stated extra
   assumptions. If it does not, the model is wrong and needs revision
   before anything else is worth doing. (Low risk, most necessary first
   step, likely 2-4 weeks of a specialist's time.)
2. **Attempt Conjecture A** for the simplest nontrivial case: a single
   Gaussian-mechanism-noised linear statistic (i.e. formalize what
   `seal/certificate.py`'s channel B already does empirically), aiming for
   a real, checked proof, not a sketch. (Moderate risk; this is the
   conjecture closest to existing tools, via standard concentration
   inequalities.)
3. **Attempt Conjecture B** for the same simplified case, to get a genuine
   achievability/converse *pair* -- a matching upper and lower bound is
   what turns "a construction" into "an optimal characterization," and is
   usually the difference between a workshop paper and a result a top
   venue takes seriously. (High risk: minimax lower bounds against an
   *adaptive, model-aware* adversary rather than a fixed pair of
   hypotheses are genuinely harder than standard two-point arguments, and
   this step may fail to produce a clean closed form at all -- a
   negative/impossibility finding here would still be publishable, just
   different from what was aimed for.)
4. **Only then** attempt Conjecture C, and only if 2-3 succeed cleanly --
   it is the least well-specified of the three and risks being a
   distraction if the core distribution-free tradeoff isn't solid yet.
5. **Validate empirically** against `seal/` and `seal_llm/`'s actual
   certificates at each stage: any proven bound should be checked against
   the Monte-Carlo detection-power and leakage numbers already measured in
   `docs/RESULTS.md` and `docs/RESULTS_LLM.md`, not just derived in the
   abstract.

## 5. Honest risk assessment

- This is realistically **months of focused work by someone with a
  research background in statistical hypothesis testing and/or
  differential privacy theory**, not something to attempt to finish in a
  single sitting -- the same caution applies to any claim that a chat
  session produced a proof of any of the above; treat such a claim as a
  draft to be checked line by line, not a result.
- The most likely failure mode is **Conjecture A turning out to be a
  straightforward corollary of existing concentration-inequality results
  once correctly formalized** -- in which case the actual contribution
  shrinks to "the formalization of the calibration-based, unknown-
  population setting itself," which is a real but much smaller claim than
  a new theorem, and should be reported as such rather than inflated.
- The most interesting possible outcome is a **clean impossibility result
  for Conjecture B** showing that no calibration-based audit can match the
  oracle-mechanism-knowledge tradeoff rate, with an explicit gap term
  depending on `m` -- that would be a genuine, citable, non-obvious
  contribution, but it is a conjecture here, not a finding.
- Before investing real effort: a proper literature search well beyond
  what this project's ad hoc web searches covered (a systematic review of
  the differential-privacy-auditing and minimax-hypothesis-testing
  literatures specifically) is a prerequisite, not an optional step --
  this document's related-work table in Section 1 should be treated as a
  starting point for that search, not as its conclusion.
