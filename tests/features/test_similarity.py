import numpy as np
import pytest

from media_rs.features.similarity import (
    compute_topk_similarity,
)


@pytest.fixture
def simple_matrix():
    # 3 vectors in 2D space
    return np.array([
        [1.0, 0.0],  # A
        [0.9, 0.1],  # B (close to A)
        [0.0, 1.0],  # C (far from A/B)
    ])


def test_compute_topk_similarity_basic(simple_matrix):
    result = compute_topk_similarity(simple_matrix, k=1)

    assert isinstance(result, dict)
    assert len(result) == 3

    for key, neighbors in result.items():
        assert isinstance(neighbors, list)
        assert len(neighbors) == 1
        nbr_id, sim = neighbors[0]
        assert isinstance(nbr_id, np.int64)
        assert isinstance(sim, float)


def test_compute_topk_similarity_similarity_range(simple_matrix):
    result = compute_topk_similarity(simple_matrix, k=2)

    for neighbors in result.values():
        for _, sim in neighbors:
            assert 0.0 <= sim <= 1.0

def test_compute_topk_similarity_excludes_self(simple_matrix):
    result = compute_topk_similarity(simple_matrix, k=2)

    for key, neighbors in result.items():
        neighbor_ids = [nbr for nbr, _ in neighbors]
        assert key not in neighbor_ids


def test_compute_topk_similarity_with_labels(simple_matrix):
    labels = ["A", "B", "C"]

    result = compute_topk_similarity(
        simple_matrix,
        index_labels=labels,
        k=1,
    )

    assert set(result.keys()) == set(labels)

    for key, neighbors in result.items():
        nbr_key, sim = neighbors[0]
        assert nbr_key in labels
        assert isinstance(sim, float)


def test_compute_topk_similarity_nearest_neighbor(simple_matrix):
    labels = ["A", "B", "C"]

    result = compute_topk_similarity(
        simple_matrix,
        index_labels=labels,
        k=1,
    )

    # A should be closest to B
    assert result["A"][0][0] == "B"

    # B should be closest to A
    assert result["B"][0][0] == "A"


def test_compute_topk_similarity_large_k(simple_matrix):
    result = compute_topk_similarity(simple_matrix, k=10)

    # Each item should have at most n-1 neighbors
    for neighbors in result.values():
        assert len(neighbors) == 2


@pytest.mark.parametrize("labels", [None, ["x", "y", "z"]])
def test_compute_topk_similarity_label_modes(simple_matrix, labels):
    result = compute_topk_similarity(simple_matrix, index_labels=labels, k=1)

    assert len(result) == 3