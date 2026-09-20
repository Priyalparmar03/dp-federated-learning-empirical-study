import numpy as np
import pytest

from fedpriv.data.partitioning import (
    partition_data,
    partition_dirichlet,
    partition_iid,
    partition_label_skew,
    partition_quantity_skew,
    partition_stats,
)


@pytest.fixture
def labels():
    rng = np.random.default_rng(0)
    return rng.integers(0, 4, size=2000)


def test_iid_partition_covers_all_samples_exactly_once(labels):
    parts = partition_iid(labels, n_clients=10)
    all_idx = np.concatenate(parts)
    assert len(all_idx) == len(labels)
    assert len(set(all_idx.tolist())) == len(labels)


def test_dirichlet_partition_covers_all_samples(labels):
    parts = partition_dirichlet(labels, n_clients=10, alpha=0.5)
    all_idx = np.concatenate(parts)
    assert sorted(all_idx.tolist()) == list(range(len(labels)))


def test_dirichlet_low_alpha_more_skewed_than_high_alpha(labels):
    low = partition_dirichlet(labels, n_clients=10, alpha=0.05, seed=1)
    high = partition_dirichlet(labels, n_clients=10, alpha=100.0, seed=1)

    def avg_label_entropy(parts):
        stats = partition_stats(labels, parts)
        probs = stats / np.clip(stats.sum(axis=1, keepdims=True), 1, None)
        ent = -(probs * np.log(probs + 1e-12)).sum(axis=1)
        return ent.mean()

    assert avg_label_entropy(low) < avg_label_entropy(high)


def test_quantity_skew_produces_unequal_client_sizes(labels):
    parts = partition_quantity_skew(labels, n_clients=10, sigma=1.0, seed=1)
    sizes = [len(p) for p in parts]
    assert max(sizes) > 2 * (sum(sizes) / len(sizes))  # meaningfully unequal
    assert sum(sizes) == len(labels)


def test_label_skew_limits_classes_per_client(labels):
    parts = partition_label_skew(labels, n_clients=10, labels_per_client=2, seed=1)
    stats = partition_stats(labels, parts)
    for row in stats:
        assert (row > 0).sum() <= 2


def test_partition_data_dispatch_matches_direct_call(labels):
    a = partition_data(labels, n_clients=5, strategy="iid", seed=7)
    b = partition_iid(labels, n_clients=5, seed=7)
    for x, y in zip(a, b):
        assert np.array_equal(x, y)


def test_partition_data_rejects_unknown_strategy(labels):
    with pytest.raises(ValueError):
        partition_data(labels, n_clients=5, strategy="not_a_strategy")
