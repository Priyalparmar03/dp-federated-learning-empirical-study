# """Predefined sweep grids — each corresponds directly to one research
# question / figure in PAPER_GUIDE.md. Passed to `runner.run_sweep`.
# """

# SWEEPS = {
#     # RQ1: privacy-utility tradeoff across a wide epsilon grid (tight -> loose)
#     "epsilon_sweep": {
#         "privacy.epsilon": [0.1, 0.5, 1, 2, 5, 10, 20, 50],
#     },
#     # RQ2: Gaussian vs Laplace at each epsilon
#     "mechanism_comparison": {
#         "privacy.mechanism": ["gaussian", "laplace"],
#         "privacy.epsilon": [0.1, 0.5, 1, 2, 5, 10, 20, 50],
#     },
#     # RQ3: effect of data heterogeneity on the privacy-utility tradeoff
#     "partition_comparison": {
#         "partition.strategy": ["iid", "dirichlet", "label_skew", "quantity_skew"],
#         "privacy.epsilon": [0.5, 2, 10, 50],
#     },
#     # RQ4: scalability / communication cost vs number of clients
#     "client_scaling": {
#         "federated.n_clients": [10, 20, 40, 80],
#     },
#     # RQ5: accountant tightness ablation (basic composition vs zCDP)
#     "accountant_comparison": {
#         "privacy.accountant": ["basic_composition", "analytic_gaussian"],
#         "privacy.epsilon": [0.1, 0.5, 1, 2, 5, 10, 20, 50],
#     },
#     # Fast smoke test for CI / sanity-checking the pipeline
#     "smoke_test": {
#         "privacy.epsilon": [1.0],
#     },
# }


"""Predefined sweep grids — each corresponds directly to one research
question / figure in PAPER_GUIDE.md. Passed to `runner.run_sweep`.
"""

SWEEPS = {
    # RQ1: privacy-utility tradeoff across a wide epsilon grid (tight -> loose)
    "epsilon_sweep": {
        "privacy.epsilon": [0.1, 0.5, 1, 2, 5, 10, 20, 50],
    },
    # RQ2: Gaussian vs Laplace at each epsilon
    "mechanism_comparison": {
        "privacy.mechanism": ["gaussian", "laplace"],
        "privacy.epsilon": [0.1, 0.5, 1, 2, 5, 10, 20, 50],
    },
    # RQ3: effect of data heterogeneity on the privacy-utility tradeoff
    "partition_comparison": {
        "partition.strategy": ["iid", "dirichlet", "label_skew", "quantity_skew"],
        "privacy.epsilon": [0.5, 2, 10, 50],
    },
    # RQ4: scalability / communication cost vs number of clients
    "client_scaling": {
        "federated.n_clients": [10, 20, 40, 80],
    },
    # RQ5: accountant tightness ablation (basic composition vs zCDP)
    "accountant_comparison": {
        "privacy.accountant": ["basic_composition", "analytic_gaussian"],
        "privacy.epsilon": [0.1, 0.5, 1, 2, 5, 10, 20, 50],
    },
    # Fast smoke test for CI / sanity-checking the pipeline
    "smoke_test": {
        "privacy.epsilon": [1.0],
    },

    # ----- multi-seed versions: same grids as above, x5 seeds, with -----
    # ----- mean/std aggregation (see experiments/runner.py)         -----

    # RQ1 (multi-seed): the main privacy-utility tradeoff table, now with
    # error bars instead of a single noisy point per epsilon.
    "epsilon_sweep_multiseed": {
        "seed": [42, 43, 44, 45, 46],
        "privacy.epsilon": [0.1, 0.5, 1, 2, 5, 10, 20, 50],
    },

    # RQ6 (new): semantic (per-class) clipping vs the existing global
    # clip, at fixed nominal sensitivity (see privacy/semantic_clipping.py
    # for why the two are directly comparable). This is the ablation for
    # the semantic-clipping contribution — the headline new result.
    "clip_mode_comparison": {
        "seed": [42, 43, 44, 45, 46],
        "privacy.clip_mode": ["global", "semantic"],
        "privacy.epsilon": [0.1, 0.5, 1, 2, 5, 10, 20, 50],
    },

    # RQ7 (new, multi-seed version of PAPER_GUIDE.md 6.1): central vs
    # distributed noise placement, run through the same config flag
    # instead of manually moving code between client.py and server.py.
    "noise_placement_comparison": {
        "seed": [42, 43, 44, 45, 46],
        "privacy.noise_placement": ["central", "distributed"],
        "privacy.epsilon": [0.5, 1, 2, 5, 10, 20],
    },

    # RQ8 (new): does opting in to subsampling-amplified noise calibration
    # meaningfully change the tradeoff at fixed nominal epsilon? Off vs on.
    "amplification_comparison": {
        "seed": [42, 43, 44, 45, 46],
        "privacy.use_subsampling_amplification": [False, True],
        "privacy.epsilon": [0.1, 0.5, 1, 2, 5, 10, 20, 50],
    },
}
