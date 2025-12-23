import pytest
import numpy as np

from src.models.content import ContentModel
from src.types.model import ContentSimilarity, IdType


@pytest.fixture
def sample_data():
    """Provide a small set of items for testing"""
    ids = [1, 2, 3]
    features = [
        "apple banana",
        "banana orange",
        "apple orange banana"
    ]
    return ids, features


def test_initialization(sample_data):
    ids, features = sample_data
    model = ContentModel(ids, features, k=2)

    # Check attributes
    assert model.ids == ids
    assert model.features == features
    assert model.k == 2
    assert model.max_features == 5000
    assert model._tfidf_matrix is None
    assert model._neighbors is None


def test_tfidf_matrix_property(sample_data):
    ids, features = sample_data
    model = ContentModel(ids, features)

    tfidf1 = model.tfidf_matrix
    tfidf2 = model.tfidf_matrix  # Should return cached

    # Check type and shape
    assert isinstance(tfidf1, np.ndarray) or hasattr(tfidf1, "toarray")
    assert tfidf1.shape[0] == len(features)

    # Cached property returns the same object
    assert tfidf1 is model._tfidf_matrix
    assert tfidf1 is tfidf2


def test_neighbors_property(sample_data):
    ids, features = sample_data
    model = ContentModel(ids, features, k=2)

    neighbors = model.neighbors

    # Check type and keys
    assert isinstance(neighbors, dict)
    assert set(neighbors.keys()) == set(ids)

    # Each item should have at most k neighbors
    for n in neighbors.values():
        assert isinstance(n, list)
        assert all(isinstance(x, tuple) and len(x) == 2 for x in n)
        assert all(isinstance(x[1], float) for x in n)  # similarity
        assert len(n) <= model.k


def test_recommend_returns_top_n(sample_data):
    ids, features = sample_data
    model = ContentModel(ids, features, k=2)

    top_recs = model.recommend(1, top_n=1)
    assert isinstance(top_recs, list)
    assert len(top_recs) == 1
    assert isinstance(top_recs[0], tuple)
    assert isinstance(top_recs[0][1], float)  # similarity


def test_recommend_with_top_n_greater_than_neighbors(sample_data):
    ids, features = sample_data
    model = ContentModel(ids, features, k=1)

    # There is only 1 neighbor
    recs = model.recommend(1, top_n=5)
    assert len(recs) == 1  # Should return all available neighbors


def test_recommend_raises_for_unknown_id(sample_data):
    ids, features = sample_data
    model = ContentModel(ids, features)

    with pytest.raises(ValueError):
        model.recommend(999, top_n=1)


def test_small_dataset_k_larger_than_items():
    ids = [1, 2]
    features = ["apple", "banana"]
    model = ContentModel(ids, features, k=10)

    # Should still work, neighbors length capped at n_samples - 1
    neighbors = model.neighbors
    for n in neighbors.values():
        assert len(n) == len(ids) - 1


def test_neighbors_are_consistent(sample_data):
    ids, features = sample_data
    model = ContentModel(ids, features)

    neighbors_first = model.neighbors
    neighbors_second = model.neighbors  # Should use cache

    assert neighbors_first is neighbors_second
