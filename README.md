# SEAL: Statistically-bounded Erasure Audit under mutuaL distrust

SEAL is a protocol for auditing "right to be forgotten" / machine-unlearning
compliance claims **from the data owner's or regulator's side**, when the
model owner is not trusted. It targets a gap the recent literature names but
does not close:

* Parameter inspection cannot certify unlearning -- a model that genuinely
  forgot a record and one that never did can have arbitrarily close (or far)
  parameters (established fact; see `docs/ALGORITHM.md` for citations).
* Behavioral / membership-inference audits powerful enough to catch a
  dishonest server *provably leak information about the retained training
  population* (Tang, Joshi & Kundu, arXiv:2606.14518) -- there is no
  accepted way to control that cost or a mechanism that lets an auditor
  choose a point on the tradeoff.
* Cryptographic alternatives (ZK proofs, TEEs, blockchain audit trails)
  trade assurance for prohibitive proof size, enclave limits, or latency
  (VERIFUL, arXiv:2510.00833).

SEAL is a statistical protocol, not a cryptographic one: it gives the data
owner (i) a **zero-privacy-cost** check using only data it already owns, and
(ii) a **second, deliberately bounded and tunable** check, with a formal
`(epsilon, delta)` accounting of exactly how much retained-set privacy that
second check is allowed to spend. See `docs/ALGORITHM.md` for the full
protocol specification, the privacy accounting, what is genuinely new here
versus reused off-the-shelf (DP accounting, membership-inference proxies,
FedAvg), and the practical pitfalls that a naive implementation falls into.

It is instantiated here as an RTBF-compliance tool for a bank consortium
running federated learning, where a regulator plays the auditor (German
Credit / `credit-g`, a real bank credit-scoring dataset), and evaluated
against the standard image-classification benchmark used throughout the
unlearning literature (MNIST) so it is compared to SOTA baselines on the same
class of benchmark those baselines were designed for.

## Results at a glance

Full tables, plots, and discussion are in `docs/RESULTS.md`; raw data in
`results/`. Headline findings from the German Credit (bank consortium RTBF)
and MNIST (standard unlearning-literature benchmark) runs:

* **The impossibility results this project targets reproduce empirically.**
  Against a server that quietly ignores the erasure request, both
  parameter-diff and an unrestricted behavioral audit perform at or near
  chance (as low as exactly 0.500 AUC on MNIST) -- neither can reliably
  tell "honestly forgot" from "kept training on everything" from the model
  alone.
* **SEAL's zero-cost channel (the data owner's own forgotten records, no
  privacy budget spent at all) already dominates both baselines** on that
  case: 0.92 detection power on German Credit, 1.00 on MNIST, at literally
  zero accounted retained-set privacy cost.
* **SEAL's tunable channel exposes the designed power/leakage tradeoff**
  against the harder, targeted-spoofing adversary the zero-cost channel
  alone cannot catch: detection power rises smoothly from ~0.53-0.54 at a
  tight epsilon budget to 1.00 as the budget is relaxed, on both datasets.
* **Honestly reported limitation:** the calibrated single-instance
  accept/reject decision is currently the weakest link, not the underlying
  statistics -- it needs more calibration replicates or a finite-sample-
  corrected threshold before it should be read as a production-ready
  certificate rather than a research prototype of the mechanism. See
  `docs/RESULTS.md` for the full account, including where the certificate's
  flag rate does and does not track the raw statistic's power.

A second, independent LLM-unlearning track (`seal_llm/`) applies the same
mutual-distrust audit idea to fine-tuned language models instead of
federated classifiers -- see "The LLM track" below and `docs/RESULTS_LLM.md`
for its own results at a glance.

## The LLM track (`seal_llm/`)

A second, independent instantiation of the same mutual-distrust idea for
LLM unlearning: a data owner watermarks its contribution (Kirchenbauer et
al., ICML 2023) before it enters training, then audits a "we forgot your
data" claim from a model owner it does not trust using a real Paillier-PIR
oblivious batch with canaries -- built specifically to survive a server
that recognizes and filters the audit check itself, a problem the closest
prior work (WaterDrum) names as open in its own Appendix D. See
`docs/ALGORITHM_LLM.md` for the protocol and `docs/RESULTS_LLM.md` for the
numbers from an end-to-end run on distilgpt2. Headline results:

* Against a server that never removes the author's influence and instead
  filters output for the one exact known audit-trigger string, a naive
  audit that only ever asks that one prompt is fooled completely -- the
  real protocol, which never uses that literal string, is not.
* Against a smarter server that recognizes the TOPIC and suppresses every
  phrasing uniformly, a slot-hiding defense alone (which query is checked)
  buys nothing -- a separate response-diversity channel is what catches it.
* Against a resource-bounded server that can only guess a subset of likely
  phrasings to fake, a real-cluster of independently-paraphrased queries
  with one checked at random turns detection into the kind of
  hypergeometric guessing game this repository can actually derive a
  formula for and check empirically.

`docs/RESULTS_LLM.md` also states plainly what this does not show,
including the sample-size and calibration-noise caveats analogous to the
classical track's `docs/RESULTS.md`.

## Layout

```
seal/
  flmodel.py       federated (FedAvg) multinomial softmax classifier, numpy-only
  mechanisms.py     honest retrain / honest continued-training / dishonest "lazy" / dishonest "spoof"
  privacy.py        Gaussian-mechanism / zCDP accounting for the bounded audit channel
  certificate.py    the SEAL two-channel certificate-of-erasure protocol
  baselines.py      parameter-diff audit; unrestricted behavioral audit (Tang et al.-style)
  metrics.py        Monte-Carlo detection power; empirical retained-set leakage (loss-based MIA proxy)
  data.py           dataset loading + population split (train / calibration / holdout pools)
  evaluate.py       experiment orchestration (Monte-Carlo trials, epsilon sweep, Pareto summary)
seal_llm/
  watermark.py      Kirchenbauer et al. green/red-list LLM watermark: biased generation + z-test detection
  pir.py            real single-server PIR over Paillier homomorphic encryption (phe)
  data.py           fictitious (TOFU-style) author profiles, watermarked documents, paraphrase templates
  model.py          distilgpt2 fine-tuning / generation utilities
  mechanisms.py     honest unlearn / lazy / naive exact-match filter / topic filter / partial topic filter
  certificate.py    SEAL-W: real-cluster + diversity-channel certificate, real PIR retrieval, calibrated decision
scripts/
  run_experiments.py     runs the classical-track comparison on both datasets, writes results/*.csv, results/*.png
  run_llm_experiment.py  runs the LLM-track SEAL-W experiment, writes results/llm_watermark_*.csv
tests/                 unit tests for the privacy accounting, certificate logic, PIR, and watermark
docs/ALGORITHM.md       classical-track formal protocol spec, novelty discussion, privacy proof, known pitfalls
docs/ALGORITHM_LLM.md   LLM-track protocol spec: threat model, novelty accounting, hypergeometric bound
docs/RESULTS_LLM.md     LLM-track results: honest numbers from an end-to-end distilgpt2 run
results/                generated CSVs and plots from the last run
```

## Running it

```
pip install -r requirements.txt
python3 -m pytest tests/ -q
python3 scripts/run_experiments.py       # classical (federated) track
python3 scripts/run_llm_experiment.py    # LLM track (fine-tunes distilgpt2 several times; CPU, ~1 hour)
```

This trains a small federated classifier on each dataset, simulates an
honest server (retrains without the forgotten records), an honest-but-cheap
server (continues training without them), a dishonest server that ignores
the request (keeps training on all the data, including what it was asked to
forget), and a dishonest server that narrowly spoofs compliance (fine-tunes
on exactly the forgotten records to fool an own-data-only check). It then
audits every (honest, dishonest) pair with SEAL across a sweep of privacy
budgets and with the two baselines, and reports:

* **detection power** -- Monte-Carlo AUC of the audit statistic separating
  honest from dishonest outcomes across many independent forget requests;
* **retained-set leakage** -- empirical membership-inference AUC of an
  attacker given exactly what the audit channel releases.

Results (`results/*_summary.csv`, `results/*_pareto.png`) are discussed in
`docs/RESULTS.md`.

## What is genuinely new vs. reused

Read `docs/ALGORITHM.md` Section 2 for the honest accounting: SEAL reuses
standard tools (the Gaussian mechanism / zCDP composition, loss-based
membership inference as a leakage proxy, FedAvg) but the *protocol* --
splitting the audit into a zero-cost own-data channel and a bounded,
self-generated boundary-probe channel, and using the shape of the induced
loss-decay (not just its mean) to catch targeted spoofing -- is a new
construction that directly answers the open problem this repository is
named after: there is currently no accepted scheme for auditing unlearning
under mutual distrust between data owner and model owner.
