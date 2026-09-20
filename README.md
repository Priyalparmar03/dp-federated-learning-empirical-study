# fedpriv — Privacy-Preserving Federated Learning with Differential Privacy

**An empirical study of the privacy-utility tradeoff under Gaussian and
Laplace Differential Privacy, across IID and non-IID client data.**

This is a from-scratch, dependency-light federated learning framework
built to be simultaneously (a) a runnable research codebase, (b) a GitHub
portfolio piece, and (c) the experimental backbone for a short paper. See
[`PAPER_GUIDE.md`](PAPER_GUIDE.md) for how to turn the results this
framework produces into an actual paper.

## Why this exists, and a design note up front

Everything here runs **today, with zero network access and zero GPU**,
on a synthetic tabular dataset generated locally. `torch`/`torchvision`
are optional — if present, they unlock CIFAR-10, Fashion-MNIST, and a
CNN model; if absent, `dataset_manager.py` transparently falls back to
the synthetic dataset with a logged warning rather than crashing. This
was a deliberate engineering choice, not a limitation to hide: a
research framework that only works in one specific environment isn't
reproducible.

## What's actually implemented (not aspirational)

| Component | Where | Notes |
|---|---|---|
| FedAvg | `fedpriv/federated/aggregation.py` | Sample-weighted (standard) or uniform/fixed-denominator (required for central DP) |
| FedProx | `fedpriv/federated/client.py` | Proximal term at the client |
| Gaussian / Laplace DP | `fedpriv/privacy/mechanisms.py` | Central DP: one noise draw per round, added to the aggregate |
| L2 clipping | `fedpriv/privacy/clipping.py` | Per-client update clipping |
| Privacy accountant | `fedpriv/privacy/accountant.py` | Two modes: naive composition, and zCDP composition (Bun & Steinke 2016) — tested against each other |
| Non-IID partitioning | `fedpriv/data/partitioning.py` | IID, Dirichlet (Hsu et al. 2019), pathological label-skew (McMahan et al. 2017), quantity-skew (log-normal) |
| Dataset manager | `fedpriv/data/dataset_manager.py` | synthetic (default) / CIFAR-10 / Fashion-MNIST / Adult, one config line to switch |
| Network simulation | `fedpriv/federated/communication.py` | Latency jitter, bandwidth-limited transmission time, random dropout |
| NumPy reference model | `fedpriv/models/mlp_numpy.py` | Hand-written forward/backward, no torch required |
| Optional PyTorch models | `fedpriv/models/torch_models.py` | MLP + CNN, same flat-parameter-vector interface |
| Metrics | `fedpriv/evaluation/metrics.py` | Accuracy, precision, recall, F1, loss, confusion matrix |
| Plots | `fedpriv/visualization/plots.py` | PNG + PDF, all the standard FL/DP paper figures |
| Sweeps | `fedpriv/experiments/sweeps.py`, `runner.py` | 6 predefined sweeps mapped 1:1 to research questions |

## Quickstart

```bash
pip install -r requirements.txt
python app.py run --config configs/default.yaml
```

Override anything from the CLI without touching YAML:

```bash
python app.py run --config configs/default.yaml \
  --set federated.n_rounds=50 privacy.epsilon=2.0 partition.strategy=label_skew
```

Run a full research sweep (the one that produced this repo's headline
figure):

```bash
python app.py sweep --config configs/default.yaml --sweep epsilon_sweep
python app.py sweep --list   # see all 6 predefined sweeps
```

Every run writes to `outputs/`:
```
outputs/
  results/<run_id>/{history.csv, summary.json, config.json, metrics.jsonl}
  plots/<run_id>/*.png, *.pdf
  checkpoints/<run_id>/final.npy
  logs/<run_id>.log
```

Run tests (pytest, or manually if pytest isn't installed):
```bash
pytest tests/ -v
```

## Verified result (already run, not hypothetical)

The headline privacy-utility tradeoff, `synthetic` dataset, Dirichlet
non-IID partition (α=0.5), 40 clients / 75% participation, Gaussian
mechanism, zCDP accounting, 15 rounds:

| ε | Test Accuracy | Test F1 | Noise σ |
|---|---|---|---|
| 0.1 | 0.222 | 0.156 | 6.208 |
| 0.5 | 0.213 | 0.162 | 1.252 |
| 1 | 0.250 | 0.200 | 0.633 |
| 2 | 0.319 | 0.276 | 0.323 |
| 5 | 0.436 | 0.429 | 0.136 |
| 10 | 0.553 | 0.551 | 0.073 |
| 20 | 0.652 | 0.652 | 0.041 |
| 50 | 0.674 | 0.673 | 0.021 |
| non-private baseline | 0.836 | 0.834 | — |

Regenerate with:
```bash
python app.py sweep --config configs/default.yaml --sweep epsilon_sweep
python app.py run --config configs/no_privacy_baseline.yaml
```

## A methodological choice worth knowing about (and citing)

DP noise is added **once, centrally, by the server, after aggregation**
— not independently by every client before averaging. This is the
"DP-FedAvg" construction of McMahan et al. (2018), *Learning
Differentially Private Recurrent Language Models*. The reason it matters:
the sensitivity of the *aggregate* to one client's data is
`clip_norm / cohort_size`, not `clip_norm` — so central DP needs far
less noise for the same (ε, δ) guarantee than naively having every
client self-noise. An earlier version of this codebase did the latter;
switching to central DP moved the ε=10 test accuracy from ~35% to ~55%
on identical data, which is itself a demonstrable, reportable finding
(see PAPER_GUIDE.md, "Ablation: central vs. distributed noise").

## Threat model / what DP guarantee this actually provides

This is **client-level** (user-level) DP: the unit of privacy protection
is "one client's entire local dataset," not "one training example." A
client's presence or absence, and the content of their local data, is
protected; DP-SGD-style *per-example* privacy within a client's own data
is a different (also valid) formulation this framework does not
implement — noted explicitly so it isn't oversold in a paper or a
portfolio review.

## Known limitations (state these in a paper — reviewers will ask)

- The privacy accountant implements naive composition and zCDP
  composition, not the tighter numerical accountants (moments
  accountant / PRV accountant) used by production libraries like
  Opacus. zCDP is still valid and standard, just not maximally tight.
- No subsampling/amplification accounting: client sampling is treated
  as arbitrary, not exploited for a tighter bound, even though it
  provably would give one. Flagged as future work.
- The synthetic dataset is the only one guaranteed reproducible without
  network access; CIFAR-10/Fashion-MNIST/Adult results require torch/
  torchvision and OpenML access respectively.
- Secure aggregation is *simulated* (the server still sees clipped,
  noised individual updates before summing) — not a cryptographic
  secure-sum protocol. Framed accurately as a network/threat simulation,
  not an implemented cryptographic primitive.

## Repository layout

```
fedpriv/
  configs/                  YAML configs (default, cifar10, no-privacy baseline)
  fedpriv/
    data/                   synthetic generator, dataset manager, partitioning
    privacy/                clipping, mechanisms, accountant
    models/                 numpy MLP (always available), torch MLP+CNN (optional)
    federated/              client, server, aggregation, network simulator
    evaluation/              metrics
    visualization/           plots
    experiments/             runner, predefined sweeps
    utils/                   config, logging, seeding
  tests/                     unit tests for privacy + partitioning
  outputs/                   generated: results, plots, checkpoints, logs
  PAPER_GUIDE.md              maps this repo's outputs onto a paper
```
