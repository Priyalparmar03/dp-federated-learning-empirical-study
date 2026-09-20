import numpy as np
import pytest

from fedpriv.models.mlp_numpy import MLPShapes
from fedpriv.privacy.semantic_clipping import (
    clip_update_semantic,
    composed_sensitivity,
    default_group_clip_norms,
    get_semantic_groups,
)


@pytest.fixture
def shapes():
    return MLPShapes(in_dim=20, hidden_dim=8, n_classes=4)


def test_semantic_groups_partition_the_vector_exactly(shapes):
    groups = get_semantic_groups(shapes)
    all_idx = np.concatenate(list(groups.values()))
    assert len(all_idx) == shapes.n_params()
    # every index appears in exactly one group (no overlap, no gaps)
    assert sorted(all_idx.tolist()) == list(range(shapes.n_params()))


def test_semantic_groups_one_per_class_plus_shared(shapes):
    groups = get_semantic_groups(shapes)
    assert set(groups.keys()) == {"shared", "class_0", "class_1", "class_2", "class_3"}


def test_default_group_clip_norms_uniform_matches_global_sensitivity(shapes):
    group_norms = default_group_clip_norms(shapes, total_clip_norm=2.0)
    assert composed_sensitivity(group_norms) == pytest.approx(2.0)


def test_default_group_clip_norms_custom_weights_still_normalize_to_total(shapes):
    group_norms = default_group_clip_norms(shapes, total_clip_norm=2.0, class_weights={0: 3.0})
    assert composed_sensitivity(group_norms) == pytest.approx(2.0)
    # class 0 should get a larger share than the other (weight-1.0) classes
    assert group_norms["class_0"] > group_norms["class_1"]


def test_clip_update_semantic_only_shrinks_over_norm_groups(shapes):
    group_norms = default_group_clip_norms(shapes, total_clip_norm=10.0)
    groups = get_semantic_groups(shapes)

    update = np.zeros(shapes.n_params(), dtype=np.float64)
    # push class_0's slice way over its budget; leave everything else at 0
    update[groups["class_0"]] = 100.0

    clipped, raw_norms = clip_update_semantic(update, shapes, group_norms)

    assert np.linalg.norm(clipped[groups["class_0"]]) == pytest.approx(group_norms["class_0"])
    # untouched groups (norm 0, under budget) are unaffected
    assert np.allclose(clipped[groups["shared"]], 0.0)
    assert np.allclose(clipped[groups["class_1"]], 0.0)
    assert raw_norms["class_0"] > group_norms["class_0"]


def test_clip_update_semantic_no_op_when_all_groups_under_norm(shapes):
    group_norms = default_group_clip_norms(shapes, total_clip_norm=10.0)
    rng = np.random.default_rng(0)
    update = rng.normal(scale=0.01, size=shapes.n_params())  # small, well under any group's budget
    clipped, _ = clip_update_semantic(update, shapes, group_norms)
    assert np.allclose(clipped, update)
