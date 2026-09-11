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
scripts/
  run_experiments.py   runs the full comparison on both datasets, writes results/*.csv, results/*.png
tests/                 unit tests for the privacy accounting and the certificate logic
docs/ALGORITHM.md       formal protocol spec, novelty discussion, privacy proof, known pitfalls
results/                generated CSVs and plots from the last run
```

## Running it

```
pip install -r requirements.txt
python3 -m pytest tests/ -q
python3 scripts/run_experiments.py
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
