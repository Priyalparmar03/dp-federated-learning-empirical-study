# """Configuration system.

# Design goal: a single flat YAML file per experiment, strongly typed via
# dataclasses, with dot-path CLI overrides (e.g. `privacy.epsilon=0.5`) so
# sweeps don't require hand-editing YAML.
# """
# from __future__ import annotations

# import copy
# from dataclasses import dataclass, field, fields, is_dataclass
# from pathlib import Path
# from typing import Any, Dict, List, Optional, get_type_hints

# import yaml


# @dataclass
# class DatasetConfig:
#     name: str = "synthetic"              # synthetic | cifar10 | fashion_mnist | adult
#     root: str = "./data"
#     n_samples: int = 6000                # synthetic only
#     n_features: int = 20                  # synthetic only
#     n_classes: int = 4                    # synthetic only
#     val_split: float = 0.15
#     test_split: float = 0.15


# @dataclass
# class PartitionConfig:
#     strategy: str = "iid"                 # iid | dirichlet | label_skew | quantity_skew
#     dirichlet_alpha: float = 0.5
#     labels_per_client: int = 2            # label_skew
#     quantity_skew_sigma: float = 0.5      # quantity_skew (log-normal sigma)


# @dataclass
# class PrivacyConfig:
#     enabled: bool = True
#     mechanism: str = "gaussian"           # gaussian | laplace | none
#     epsilon: float = 1.0
#     delta: float = 1e-5
#     clip_norm: float = 1.0                # L2 clipping bound C
#     accountant: str = "analytic_gaussian"  # analytic_gaussian | basic_composition


# @dataclass
# class NetworkConfig:
#     simulate: bool = True
#     base_latency_ms: float = 20.0
#     latency_jitter_ms: float = 15.0
#     bandwidth_mbps: float = 10.0
#     dropout_prob: float = 0.05
#     asynchronous: bool = False


# @dataclass
# class FederatedConfig:
#     algorithm: str = "fedavg"             # fedavg | fedprox
#     fedprox_mu: float = 0.01
#     n_clients: int = 10
#     client_fraction: float = 0.6
#     n_rounds: int = 30
#     local_epochs: int = 2
#     batch_size: int = 32
#     learning_rate: float = 0.05


# @dataclass
# class ExperimentConfig:
#     run_id: str = "run"
#     seed: int = 42
#     output_dir: str = "outputs"
#     dataset: DatasetConfig = field(default_factory=DatasetConfig)
#     partition: PartitionConfig = field(default_factory=PartitionConfig)
#     privacy: PrivacyConfig = field(default_factory=PrivacyConfig)
#     network: NetworkConfig = field(default_factory=NetworkConfig)
#     federated: FederatedConfig = field(default_factory=FederatedConfig)


# def _dict_to_dataclass(cls, data: Dict[str, Any]):
#     # `from __future__ import annotations` (used throughout this module)
#     # turns dataclass field.type into a STRING, so is_dataclass(f.type)
#     # would silently be False. get_type_hints() resolves the real classes.
#     hints = get_type_hints(cls)
#     kwargs = {}
#     for f in fields(cls):
#         if f.name not in data:
#             continue
#         val = data[f.name]
#         real_type = hints.get(f.name, f.type)
#         if is_dataclass(real_type) and isinstance(val, dict):
#             kwargs[f.name] = _dict_to_dataclass(real_type, val)
#         else:
#             kwargs[f.name] = val
#     return cls(**kwargs)


# def load_config(path: Optional[str] = None, overrides: Optional[List[str]] = None) -> ExperimentConfig:
#     """Load an ExperimentConfig from YAML, applying dot-path overrides.

#     overrides: list of "a.b.c=value" strings (as passed on the CLI).
#     """
#     raw: Dict[str, Any] = {}
#     if path is not None and Path(path).exists():
#         with open(path) as fh:
#             raw = yaml.safe_load(fh) or {}

#     cfg = _dict_to_dataclass(ExperimentConfig, raw)

#     for ov in overrides or []:
#         key, _, value = ov.partition("=")
#         _apply_override(cfg, key.split("."), _parse_scalar(value))

#     return cfg


# def _apply_override(obj: Any, path: List[str], value: Any) -> None:
#     attr = path[0]
#     if len(path) == 1:
#         current = getattr(obj, attr)
#         target_type = type(current)
#         if target_type in (int, float, bool, str):
#             value = target_type(value) if not isinstance(value, bool) else value
#         setattr(obj, attr, value)
#     else:
#         _apply_override(getattr(obj, attr), path[1:], value)


# def _parse_scalar(s: str) -> Any:
#     s = s.strip()
#     if s.lower() in ("true", "false"):
#         return s.lower() == "true"
#     try:
#         if "." in s or "e" in s.lower():
#             return float(s)
#         return int(s)
#     except ValueError:
#         return s


# def config_to_dict(cfg: ExperimentConfig) -> Dict[str, Any]:
#     def _rec(obj):
#         if is_dataclass(obj):
#             return {f.name: _rec(getattr(obj, f.name)) for f in fields(obj)}
#         return obj

#     return copy.deepcopy(_rec(cfg))



"""Configuration system.

Design goal: a single flat YAML file per experiment, strongly typed via
dataclasses, with dot-path CLI overrides (e.g. `privacy.epsilon=0.5`) so
sweeps don't require hand-editing YAML.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, get_type_hints

import yaml


@dataclass
class DatasetConfig:
    name: str = "synthetic"              # synthetic | cifar10 | fashion_mnist | adult
    root: str = "./data"
    n_samples: int = 6000                # synthetic only
    n_features: int = 20                  # synthetic only
    n_classes: int = 4                    # synthetic only
    val_split: float = 0.15
    test_split: float = 0.15


@dataclass
class PartitionConfig:
    strategy: str = "iid"                 # iid | dirichlet | label_skew | quantity_skew
    dirichlet_alpha: float = 0.5
    labels_per_client: int = 2            # label_skew
    quantity_skew_sigma: float = 0.5      # quantity_skew (log-normal sigma)


@dataclass
class PrivacyConfig:
    enabled: bool = True
    mechanism: str = "gaussian"           # gaussian | laplace | none
    epsilon: float = 1.0
    delta: float = 1e-5
    clip_norm: float = 1.0                # L2 clipping bound C
    accountant: str = "analytic_gaussian"  # analytic_gaussian | basic_composition

    # --- clipping strategy ---
    clip_mode: str = "global"             # global | semantic
    # Optional per-group weighting for semantic clipping, e.g. {0: 2.0} gives
    # class 0 twice the relative clip budget of other classes; "shared" key
    # weights the trunk (W1, b1) group. None = uniform weighting (default),
    # which makes semantic clipping's TOTAL sensitivity identical to a
    # global clip of `clip_norm` (see privacy/semantic_clipping.py).
    semantic_class_weights: Optional[Dict] = None

    # --- noise placement ---
    noise_placement: str = "central"      # central | distributed
    # central: ONE noise draw added to the server-side aggregate (standard
    #   DP-FedAvg, McMahan et al. 2018).
    # distributed: every client adds its OWN independent noise draw,
    #   calibrated to full (undivided) sensitivity, before transmission.
    #   Included as an ablation baseline — see PAPER_GUIDE.md 6.1.

    # --- privacy amplification by subsampling ---
    # Off by default so existing single-run results stay exactly
    # reproducible; turn on explicitly to fold the free privacy gained by
    # only sampling `federated.client_fraction` of clients each round into
    # the noise calibration (see privacy/accountant.py for the math and
    # its documented approximation caveat).
    use_subsampling_amplification: bool = False


@dataclass
class NetworkConfig:
    simulate: bool = True
    base_latency_ms: float = 20.0
    latency_jitter_ms: float = 15.0
    bandwidth_mbps: float = 10.0
    dropout_prob: float = 0.05
    asynchronous: bool = False


@dataclass
class FederatedConfig:
    algorithm: str = "fedavg"             # fedavg | fedprox
    fedprox_mu: float = 0.01
    n_clients: int = 10
    client_fraction: float = 0.6
    n_rounds: int = 30
    local_epochs: int = 2
    batch_size: int = 32
    learning_rate: float = 0.05


@dataclass
class ExperimentConfig:
    run_id: str = "run"
    seed: int = 42
    output_dir: str = "outputs"
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    partition: PartitionConfig = field(default_factory=PartitionConfig)
    privacy: PrivacyConfig = field(default_factory=PrivacyConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    federated: FederatedConfig = field(default_factory=FederatedConfig)


def _dict_to_dataclass(cls, data: Dict[str, Any]):
    # `from __future__ import annotations` (used throughout this module)
    # turns dataclass field.type into a STRING, so is_dataclass(f.type)
    # would silently be False. get_type_hints() resolves the real classes.
    hints = get_type_hints(cls)
    kwargs = {}
    for f in fields(cls):
        if f.name not in data:
            continue
        val = data[f.name]
        real_type = hints.get(f.name, f.type)
        if is_dataclass(real_type) and isinstance(val, dict):
            kwargs[f.name] = _dict_to_dataclass(real_type, val)
        else:
            kwargs[f.name] = val
    return cls(**kwargs)


def load_config(path: Optional[str] = None, overrides: Optional[List[str]] = None) -> ExperimentConfig:
    """Load an ExperimentConfig from YAML, applying dot-path overrides.

    overrides: list of "a.b.c=value" strings (as passed on the CLI).
    """
    raw: Dict[str, Any] = {}
    if path is not None and Path(path).exists():
        with open(path) as fh:
            raw = yaml.safe_load(fh) or {}

    cfg = _dict_to_dataclass(ExperimentConfig, raw)

    for ov in overrides or []:
        key, _, value = ov.partition("=")
        _apply_override(cfg, key.split("."), _parse_scalar(value))

    return cfg


def _apply_override(obj: Any, path: List[str], value: Any) -> None:
    attr = path[0]
    if len(path) == 1:
        current = getattr(obj, attr)
        target_type = type(current)
        if target_type in (int, float, bool, str):
            value = target_type(value) if not isinstance(value, bool) else value
        setattr(obj, attr, value)
    else:
        _apply_override(getattr(obj, attr), path[1:], value)


def _parse_scalar(s: str) -> Any:
    s = s.strip()
    if s.lower() in ("true", "false"):
        return s.lower() == "true"
    try:
        if "." in s or "e" in s.lower():
            return float(s)
        return int(s)
    except ValueError:
        return s


def config_to_dict(cfg: ExperimentConfig) -> Dict[str, Any]:
    def _rec(obj):
        if is_dataclass(obj):
            return {f.name: _rec(getattr(obj, f.name)) for f in fields(obj)}
        return obj

    return copy.deepcopy(_rec(cfg))
