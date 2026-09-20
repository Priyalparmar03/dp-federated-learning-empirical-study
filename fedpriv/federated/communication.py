"""Simulates the network layer between server and clients: latency jitter,
bandwidth-limited transmission time, and random client/update dropout —
so the framework produces real "communication cost" and "client
participation rate" metrics instead of pretending FL happens over an
instantaneous, lossless channel.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fedpriv.utils.config import NetworkConfig

BYTES_PER_PARAM = 4  # float32


@dataclass
class TransmissionResult:
    delivered: bool
    latency_ms: float
    transmission_time_ms: float
    payload_bytes: int


class NetworkSimulator:
    def __init__(self, cfg: NetworkConfig, seed: int = 42):
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)

    def payload_bytes(self, n_params: int) -> int:
        return n_params * BYTES_PER_PARAM

    def simulate_upload(self, n_params: int) -> TransmissionResult:
        payload = self.payload_bytes(n_params)

        if not self.cfg.simulate:
            return TransmissionResult(True, 0.0, 0.0, payload)

        dropped = self.rng.random() < self.cfg.dropout_prob
        latency = max(0.0, self.rng.normal(self.cfg.base_latency_ms, self.cfg.latency_jitter_ms))
        bandwidth_bytes_per_ms = (self.cfg.bandwidth_mbps * 1_000_000 / 8) / 1000.0
        transmission_time = payload / max(bandwidth_bytes_per_ms, 1e-6)

        return TransmissionResult(
            delivered=not dropped,
            latency_ms=latency,
            transmission_time_ms=transmission_time,
            payload_bytes=payload,
        )
