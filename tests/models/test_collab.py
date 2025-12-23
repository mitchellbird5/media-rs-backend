import pytest
import numpy as np

from models.collab import CollaborativeModel
from rs_types.model import CollabMethod, ContentSimilarity, IdType


@pytest.fixture
def small_user_item_matrix():
    # 3 users x 3 items
    # Rows: items, Columns: users
    return np.array([
        [5, 3, 0],   # Item 1
        [4, 0, 0],   # Item 2
        [1, 1, 0]    # Item 3
    ])


@pytest.fixture
def item_ids():
    return [101, 102, 103]


def test_initialization(item_ids, small_user_item_matrix):
    model = CollaborativeModel(
        ids=item_ids,
        user_item_matrix=small_user_item_matrix,
        collab_method=CollabMethod.ITEM,
        k=2
    )

    assert model.ids == item_ids
    assert model.k == 2
    assert model.user_item_matrix is small_user_item_matrix
    assert model.collab_method == CollabMethod.ITEM
    assert model._neighbors is None


@pytest.mark.parametrize("method", [CollabMethod.ITEM, CollabMethod.USER])
def test_neighbors_property(item_ids, small_user_item_matrix, method):
    model = CollaborativeModel(
        ids=item_ids,
        user_item_matrix=small_user_item_matrix,
        collab_method=method,
        k=2
    )

    neighbors = model.neighbors

    # Check type and keys
    assert isinstance(neighbors, dict)
    assert set(neighbors.keys()) == set(item_ids)

    # Each neighbor list is correct
    for n in neighbors.values():
        assert isinstance(n, list)
        assert all(isinstance(x, tuple) and len(x) == 2 for x in n)
        assert all(isinstance(x[1], float) for x in n)
        # Length should be <= k or n_samples-1
        assert len(n) <= model.k


def test_recommend_returns_top_n(item_ids, small_user_item_matrix):
    model = CollaborativeModel(
        ids=item_ids,
        user_item_matrix=small_user_item_matrix,
        collab_method=CollabMethod.ITEM,
        k=2
    )

    top_recs = model.recommend(item_ids[0], top_n=1)
    assert isinstance(top_recs, list)
    assert len(top_recs) == 1
    assert isinstance(top_recs[0], tuple)
    assert isinstance(top_recs[0][1], float)


def test_recommend_top_n_greater_than_neighbors(item_ids, small_user_item_matrix):
    model = CollaborativeModel(
        ids=item_ids,
        user_item_matrix=small_user_item_matrix,
        collab_method=CollabMethod.ITEM,
        k=len(item_ids) - 1
    )

    recs = model.recommend(item_ids[0], top_n=5)
    # Should return all available neighbors (n_samples - 1)
    assert len(recs) == len(item_ids) - 1


def test_recommend_raises_for_unknown_id(item_ids, small_user_item_matrix):
    model = CollaborativeModel(
        ids=item_ids,
        user_item_matrix=small_user_item_matrix,
        collab_method=CollabMethod.ITEM,
        k=2
    )

    with pytest.raises(ValueError):
        model.recommend(999, top_n=1)