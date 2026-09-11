# SEAL: Statistically-bounded Erasure Audit under mutuaL distrust

This document specifies the algorithm implemented in `seal/`. It is written
so a reader can judge exactly which parts are novel and which parts are
standard tools reused for a new purpose.

## 1. Problem

A data owner (or a regulator acting on their behalf) submitted an erasure
request for a record set `D_f` that was used to train a federated model.
The model owner ("the server") returns a new model `M'` and claims `D_f`
has been unlearned. The data owner does not trust the server. Two facts
make this hard:

1. **Parameter collision.** `M'` can be numerically close to, or far from,
   a model that genuinely forgot `D_f` -- parameters alone are consistent
   with both an honest and a dishonest server (Tang, Joshi & Kundu 2026,
   arXiv:2606.14518 gives the impossibility argument for parameter-level
   checks and for weak behavioral checks).
2. **Behavioral audits leak.** Tang et al.'s "geometric transfer theorem"
   shows that any behavioral (query-only) audit accurate enough to catch a
   dishonest server necessarily reveals information about which records
   are in the retained set `D_r = D \ D_f`. There is no free lunch: more
   audit power costs more retained-set privacy. That paper characterizes
   the tradeoff; it does not give a mechanism for choosing a point on it,
   and it is not evaluated in the federated, mutual-distrust setting
   sketched by the VERIFUL framework (Nguyen et al. 2025, arXiv:2510.00833)
   or against a real, targeted forgery.

## 2. What is new here and what is not

Standard, off-the-shelf components used as-is:

* Gaussian mechanism / zero-concentrated differential privacy (zCDP)
  accounting for composing many queries (`seal/privacy.py`).
* Loss-based membership inference as the empirical proxy for "how much did
  this query leak" (`seal/metrics.py`).
* FedAvg for the federation itself (`seal/flmodel.py`).
* Two-sample/one-sample z-tests against an empirically calibrated null.

The new construction is the protocol built from them, specifically:

* **A zero-marginal-cost audit channel.** The data owner already possesses
  `D_f` in full. Re-evaluating `M` and `M'` on `D_f` costs no retained-set
  privacy at all -- no existing audit scheme in this space treats
  "the requester's own data" as a first-class, free audit signal; prior
  work goes straight to behavioral tests over the retained population.
* **Self-generated boundary probes as the only source of bounded
  leakage.** Instead of querying real retained records (which is what
  makes Tang et al.'s theorem bite), the data owner perturbs *its own*
  forgotten records into nearby synthetic points. This is the mechanism
  that turns "any powerful audit leaks retained-set membership" into "you
  can dial exactly how much it leaks," because the query set is
  constructed by the auditor, not drawn from someone else's data, and the
  Gaussian mechanism's sensitivity bound (Section 4) applies to it
  directly.
* **A decay-consistency test for targeted spoofing.** A server that fine-
  tunes on exactly the forgotten points can fool the zero-cost channel
  (Section 3, Channel A) by construction. It cannot fool the boundary-probe
  channel, because honest retraining moves the loss on the *neighborhood*
  of `D_f` too (an influence-function argument, Koh & Liang 2017) while a
  narrowly targeted fine-tune does not. Testing for this shape -- not just
  a raw mean-loss shift -- is the actual detection mechanism for the
  adversary the zero-cost channel is blind to.
* **A calibration procedure that reproduces the real training history.**
  Getting the null distribution right requires simulating "this same
  amount of data, baked into training from round zero, then honestly
  removed" against a disjoint public reference corpus -- not warm-starting
  a converged model, which under-estimates the honest effect size by an
  order of magnitude in our own experiments (see `seal/certificate.py`
  docstring and the debugging trail in this repository's history).

None of this claims to invent differential privacy, membership inference,
or FedAvg. The claim is the same kind of novelty as, say, DP-SGD: a new
composition of known primitives into a protocol that solves a stated open
problem (mutual-distrust erasure audit with a formal, tunable
privacy-audit tradeoff) that no prior scheme in this literature proposes.

## 3. Protocol

Notation: model before the request `M` (params `theta`), model after `M'`
(params `theta'`), forgotten records `D_f = {(x_i, y_i)}`, per-sample loss
`l(theta, x, y)`.

**Channel A (zero leakage).** The data owner computes

```
Lambda_A = mean_{(x,y) in D_f} [ l(theta', x, y) - l(theta, x, y) ]
```

entirely on its own data. No noise, no privacy cost, no trust required.

**Channel B (tunable leakage).** The data owner draws `B` synthetic probes
`x_i + eps_j` for `eps_j ~ N(0, jitter^2 I)`, `(x_i, y_i) in D_f`, and asks
the server to evaluate `M'` on them (label held fixed at `y_i`, a
label-preserving small-perturbation assumption). The server (or an honest
protocol implementation) returns the per-probe losses, clipped to
`[0, L_max]`, each perturbed by the Gaussian mechanism at noise level
`sigma`. The data owner computes

```
Lambda_B = mean_j [ noisy_l(theta', probe_j) - l(theta, probe_j) ]
```

**Calibration.** Both statistics are compared against a null distribution
built by literally re-running the honest procedure `R` times: a fresh
federation is retrained from scratch with a same-size subset of a public,
disjoint calibration corpus standing in for `D_f`, using the same base
random seed and client ordering as the real training run (this detail
matters: it is what makes the null's effect size match the real one -- see
Section 5), and then honestly unlearned the same way the server is
supposed to. This gives empirical `(mu_A, sigma_A)` and `(mu_B, sigma_B)`.

**Decision.** With target false-accusation rate `beta` (Bonferroni-split
`beta/2` per channel):

```
tau_A = mu_A - z_{1-beta/2} * sigma_A
tau_B = mu_B - z_{1-beta/2} * sigma_B
flag_dishonest  <=>  Lambda_A < tau_A  OR  Lambda_B < tau_B
```

The certificate returned to the data owner / regulator states: accept or
reject, the epsilon actually spent on channel B, and the false-accusation
rate `beta` the decision controls for.

## 4. Privacy accounting for channel B

Each probe's released value lies in `[0, L_max]` after clipping, so the
global sensitivity of one query is `L_max`. Composing `B` independent
Gaussian-mechanism queries with per-query noise `sigma`:

```
rho_per_query = L_max^2 / (2 sigma^2)          (zCDP of one Gaussian query)
rho_total      = B * rho_per_query              (sequential composition)
epsilon(delta) = rho_total + 2 sqrt(rho_total * ln(1/delta))   (Bun & Steinke 2016, Prop 3.3)
```

Given a target `(epsilon, delta)`, `seal/privacy.py::calibrate_sigma`
inverts this to the noise level `sigma` that spends exactly that budget
over `B` queries -- this is the tunable knob converting Tang et al.'s
impossibility statement into a mechanism: choose `epsilon` (how much
retained-set privacy this audit run may cost), get back `sigma`
(concretely, how loud the server's responses must be).

**Caveat, stated plainly:** this bound is over the *query-response
channel* (bounded-output-range sensitivity), not a training-time DP
guarantee about the model itself. It is the right formal object for "how
much does releasing these specific noisy numbers leak," which is exactly
what channel B releases; it does not by itself imply the trained model is
differentially private in any broader sense.

## 5. Practical pitfalls this implementation had to resolve

Documented here because they are easy to get wrong and silently produce a
certificate that never flags anything:

1. **Warm-starting the calibration replicate onto an already-converged
   model under-integrates the calibration subset.** A few extra FedAvg
   rounds barely move a converged model's decision boundary, so the
   resulting null is degenerate (mean near zero, entirely driven by noise
   floor). Fix: retrain the calibration replicate from scratch, the same
   number of rounds as the real model, with the calibration subset baked
   in from initialization.
2. **A different random seed per calibration replicate adds spurious
   trajectory variance.** Different SGD trajectories reach different local
   optima with different sensitivities to any specific held-out chunk, and
   this variance can dominate the actual "was this data forgotten" signal.
   Fix: reuse the real run's base training seed and per-client ordering
   for the replicate's from-scratch retrain; let only the honest-unlearning
   continuation step (and the choice of which calibration records stand in
   for `D_f`) vary across replicates.
3. **The "lazy dishonest" baseline must not be a literal no-op.** Freezing
   parameters (plus cosmetic float noise) is trivially caught by parameter
   distance, which does not test the claim the literature actually makes.
   The adversary of interest keeps training on data that still includes
   the supposedly-forgotten records -- a parameter shift of the same kind
   and magnitude as honest unlearning -- which is the case parameter
   inspection genuinely cannot resolve.
