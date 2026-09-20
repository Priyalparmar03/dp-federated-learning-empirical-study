# From Codebase to Paper

This document maps `fedpriv` directly onto a short empirical paper —
what to write in each section, which command produces which figure/table,
and what a reviewer will push back on. Target venue: a workshop paper /
arXiv preprint / EMAI application writing sample, ~6-8 pages.

**Working title:** *Privacy-Utility Tradeoff in Federated Learning: An
Empirical Study of Differential Privacy Mechanisms Under IID and
Non-IID Data*

---

## 1. Abstract (150-200 words)

Template — fill in the bracketed parts after running the sweeps:

> Federated learning (FL) enables collaborative model training without
> centralizing raw data, but exchanged model updates can still leak
> information about clients' data. We present `fedpriv`, an open-source
> framework combining FedAvg-style federated training with central
> Differential Privacy (DP), and use it to conduct a controlled empirical
> study of the privacy-utility tradeoff. We implement and compare Gaussian
> and Laplace noise mechanisms under two composition accountants (naive
> sequential composition and zero-Concentrated DP), across four data
> partitioning strategies (IID, Dirichlet, pathological label-skew, and
> quantity-skew) that simulate realistic non-IID client heterogeneity. On
> a controlled synthetic benchmark, we find that [test accuracy improves
> monotonically from X% at ε=0.1 to Y% at ε=50, compared to a Z%
> non-private baseline], and that [non-IID partitioning strategy S incurs
> the largest accuracy penalty at fixed ε]. We further show that
> **central** DP-FedAvg (noise added once to the aggregate, sensitivity
> scaled by cohort size) substantially outperforms naive **distributed**
> DP (independent per-client noise) at identical (ε, δ) — a
> reproducible ablation with direct implications for DP-FL system design.

Fill in X/Y/Z from `outputs/results/epsilon_sweep/comparison_table.csv`
and `outputs/results/no_privacy_baseline/summary.json`.

---

## 2. Introduction

Structure (each ~1 paragraph):
1. FL motivation — collaborative training without centralizing data.
2. The gap: model updates are not automatically private (cite the
   membership-inference / gradient-inversion literature generically —
   don't fabricate specific numbers from papers you haven't read).
3. DP as the standard mitigation; the central question — how much
   utility do you give up, and does that cost depend on *how*
   heterogeneous your clients are?
4. Contributions (bullet list — this is literally the feature table in
   README.md, restated as claims):
   - An open, reproducible framework unifying FedAvg + central DP + 4
     non-IID partitioning strategies + 2 privacy accountants.
   - A controlled empirical study spanning 8 epsilon values × 2
     mechanisms × 4 partition strategies.
   - An ablation isolating the accounting method's practical impact
     (naive composition vs. zCDP) and the aggregation method's impact
     (central vs. distributed noise).

## 3. Related Work (short, don't overclaim)

Three paragraphs, each citing the *foundational* papers by name (you
should actually read at least the abstracts before citing):
- **Federated Learning**: McMahan et al. 2017 (FedAvg), Li et al. 2020
  (FedProx, non-IID robustness).
- **Differential Privacy**: Dwork & Roth 2014 (survey/textbook),
  Abadi et al. 2016 (DP-SGD), Bun & Steinke 2016 (zCDP).
- **DP + FL together**: McMahan et al. 2018 (DP-FedAvg, the specific
  central-noise construction this framework implements), Kairouz et al.
  2021 (FL survey, has a DP section).

Position this work: not a new mechanism, but a *reproducible empirical
study* + a framework other students/researchers can extend. Say that
plainly — don't claim novelty you don't have.

## 4. Method

### 4.1 Federated Learning Protocol
FedAvg with client subsampling. Give the round structure as a numbered
list or a diagram (see below) — draw it once, e.g. with the `diagram`
mode of any diagramming tool, or a simple TikZ/matplotlib figure:
broadcast → local SGD → clip → central aggregate → add noise → evaluate.

### 4.2 Differential Privacy
State the exact mechanism: L2 clipping at the client (bound C), central
Gaussian/Laplace noise at the server calibrated to sensitivity
`C / cohort_size`. Give the accountant math from
`fedpriv/privacy/accountant.py`'s docstring almost verbatim — it's
already written in paper-appropriate language (zCDP composition,
Bun & Steinke inversion formula). This is your Section 4.2 nearly for
free; adapt notation to match your paper's conventions.

**State the threat model explicitly** (client-level DP, not per-example
DP-SGD) — this is exactly the "Threat model" paragraph in README.md.

### 4.3 Non-IID Partitioning
Describe all four strategies in 1-2 sentences each (the docstrings in
`fedpriv/data/partitioning.py` are already written at this level).
Cite Hsu et al. 2019 for Dirichlet partitioning specifically.

### 4.4 Datasets and Model
Synthetic tabular benchmark (`sklearn.make_classification`, N features,
K classes, controllable separability) as the primary controlled
benchmark — justify this choice explicitly: it lets you *isolate* the
DP/non-IID effects without confounding from a specific real dataset's
idiosyncrasies, which is exactly what a controlled empirical study
needs. Mention CIFAR-10/Fashion-MNIST/Adult as supported-but-optional
extensions (name them, note they require torch/torchvision/network,
and that results in the paper are on the synthetic benchmark unless you
actually ran them — don't claim results you didn't generate).

Model: 1-hidden-layer MLP (state exact dims from
`fedpriv/federated/server.py`, currently hidden_dim=8 — mention this was
chosen specifically to keep the parameter count low relative to the
achievable per-round privacy budget, and say why in 4.5).

### 4.5 A note on dimensionality (a real finding, include it)
Client-level DP noise is added to *every* model parameter each round,
so total noise power scales with the parameter count d while the signal
is bounded by clip_norm regardless of d. This means, for fixed (ε, δ,
rounds, cohort size), there is a real capacity/privacy tradeoff: smaller
models achieve better utility at the same privacy budget. This is worth
one paragraph and (optionally) one small additional experiment varying
hidden_dim, since it's a genuine, reproducible, citable observation this
codebase surfaces (see `fedpriv/federated/server.py`'s comment on this).

---

## 5. Experimental Setup

State exactly, as a table:

| Parameter | Value |
|---|---|
| Clients | 40 |
| Client participation fraction | 0.75 (cohort size 30) |
| Communication rounds | 15 |
| Local epochs | 2 |
| Batch size | 32 |
| Learning rate | 0.05 |
| Clip norm C | 1.0 |
| δ | 1e-5 |
| Partition (default) | Dirichlet, α=0.5 |
| Seed | 42 |

State that all numbers are reproducible with the exact commands in
Section 6 below — this IS your reproducibility statement, reviewers
love this.

## 6. Results — map each figure/table to the exact command

Run these once, in order, then pull numbers/figures straight out of
`outputs/`:

```bash
python app.py sweep --config configs/default.yaml --sweep epsilon_sweep
python app.py sweep --config configs/default.yaml --sweep mechanism_comparison
python app.py sweep --config configs/default.yaml --sweep partition_comparison
python app.py sweep --config configs/default.yaml --sweep accountant_comparison
python app.py sweep --config configs/default.yaml --sweep client_scaling
python app.py run    --config configs/no_privacy_baseline.yaml
```

| Paper artifact | Source file |
|---|---|
| **Fig. 1** Privacy-utility tradeoff (main result) | `outputs/plots/epsilon_sweep/privacy_vs_accuracy.pdf` |
| **Fig. 2** Accuracy vs. round, by epsilon | `outputs/plots/epsilon_sweep/accuracy_vs_rounds_by_epsilon.pdf` |
| **Fig. 3** Gaussian vs. Laplace | `outputs/plots/mechanism_comparison/gaussian_vs_laplace.pdf` |
| **Fig. 4** IID vs. non-IID | `outputs/plots/partition_comparison/iid_vs_noniid.pdf` |
| **Fig. 5** Accountant ablation (basic vs. zCDP) | `outputs/plots/accountant_comparison/privacy_vs_accuracy.pdf` |
| **Fig. 6** Communication cost vs. client count | `outputs/plots/client_scaling/communication_cost.pdf` |
| **Fig. 7** Confusion matrix (representative run) | generate via the snippet in README / re-run at your chosen ε |
| **Table 1** Main comparison table | `outputs/results/epsilon_sweep/comparison_table.csv` |
| **Table 2** Non-IID comparison | `outputs/results/partition_comparison/comparison_table.csv` |
| **Table 3** Central vs. distributed noise ablation | run both variants (see 6.1 below) and hand-build |

Write 1 paragraph per figure: state the trend in words, then the
specific numbers, then one sentence of interpretation. Don't just
caption the figure — every figure needs prose that a reader could
understand without looking at the plot.

### 6.1 Ablation: central vs. distributed noise
This is your most interesting/defensible result because you *personally
debugged it* — that's a genuine research narrative ("naive per-client
noising gives ~35% test accuracy at ε=10; switching to central DP-FedAvg
noise, holding everything else fixed, raises it to ~55%"). To reproduce
the distributed variant for the comparison, you'd temporarily move the
noise application in `federated/client.py` back before aggregation (git
diff this deliberately, keep both versions in the repo history) — this
one paragraph, with a before/after number, is a strong "we found and
fixed a real design flaw" story for an application essay or paper
discussion section.

## 7. Discussion

- Restate the core tradeoff finding in plain language.
- Discuss the dimensionality effect (Section 4.5) as a design
  implication: production DP-FL systems need either very large cohorts
  or compressed/low-rank updates to make DP practical at high dimension
  — cite this as *why* real systems (e.g., Google's Gboard DP-FL) use
  cohorts in the thousands, not tens.
- Discuss non-IID findings: does heterogeneity compound with privacy
  noise, or are they roughly independent in your results? State
  whichever your Table 2 actually shows — don't guess.

## 8. Limitations (copy from README.md, expand each to 1-2 sentences)
- Accountant is zCDP, not the tightest known (moments/PRV accountant).
- No subsampling amplification in the accounting.
- Only the synthetic benchmark was run for the reported numbers unless
  you also ran CIFAR-10/Fashion-MNIST/Adult (be honest about which).
- Secure aggregation is simulated at the network level, not
  cryptographically implemented.
- Single-machine simulation: wall-clock training time numbers reflect
  simulated network delay, not real distributed deployment.

## 9. Conclusion
2-3 sentences: what you built, what you found, what's next (e.g.
"extending to per-example DP-SGD within clients", "implementing
subsampling amplification for a tighter accountant", "evaluating on the
image datasets at GPU scale").

## 10. Reproducibility Statement
"All code, configs, and the exact commands to regenerate every figure
and table in this paper are available at [your GitHub URL]. All
experiments use a fixed seed (42) for exact reproducibility on the
synthetic benchmark; results in Section 6 were generated with
[git commit hash] of the codebase."

---

## Converting to LaTeX / IEEE / Springer format

1. Use the standard IEEEtran or Springer `llncs`/`sn-jnl` template
   (search "IEEE conference LaTeX template" or the venue's own author
   guidelines — don't guess a template structure).
2. Figures: use the `.pdf` versions from `outputs/plots/` directly —
   they're already vector format at print resolution.
3. Tables: `pandas.read_csv(...).to_latex(index=False, float_format="%.3f")`
   on any `comparison_table.csv` gives you a working LaTeX table to
   hand-tune.
4. Keep the paper to what you actually ran — every number in the paper
   should trace to a file in `outputs/`. This is the single biggest
   thing that separates a portfolio-credible paper from one that gets
   torn apart in review or an interview.

## For an Erasmus Mundus EMAI / MS-PhD application specifically

Frame this project in your statement of purpose as: "I built and
debugged a federated learning + differential privacy framework from
scratch, including implementing and comparing two DP composition
accountants, and identified and fixed a real design flaw (client- vs.
server-side noise addition) that changed measured utility by ~20
percentage points at fixed privacy budget." That's a concrete,
verifiable, specific claim — far stronger than "I built an FL/DP
project" and it maps directly onto the actual work documented in this
repo and its git history.
