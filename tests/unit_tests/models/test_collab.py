# tests/test_collaborative_models.py
import pytest
import numpy as np
import faiss
from scipy.sparse import csr_matrix

from media_rs.serving.recommender.models.collab import ItemItemCollaborativeModel, UserCollaborativeModel

# Define ContentSimilarity for testing
ContentSimilarity = tuple[int, float]


@pytest.fixture
def mock_item_item_model():
    topk_graph = {
        0: [(1, 0.9), (2, 0.8)],
        1: [(0, 0.9), (2, 0.7)]
    }
    return ItemItemCollaborativeModel(topk_graph)


def test_item_item_recommend(mock_item_item_model: ItemItemCollaborativeModel):
    # Normal case
    res = mock_item_item_model.recommend(0, 1)
    assert res == [(1, 0.9)]
    
    # top_n > available
    res = mock_item_item_model.recommend(1, 5)
    assert res == [(0, 0.9), (2, 0.7)]


@pytest.fixture
def mock_user_collaborative_model():
    # small FAISS index with 2 users, embedding_dim=3
    user_embeddings = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0]
    ], dtype=np.float32)
    faiss.normalize_L2(user_embeddings)
    
    index = faiss.IndexFlatIP(3)  # inner product = cosine for normalized vectors
    index.add(user_embeddings)
    
    # user-item matrix: 2 users x 3 items
    user_item_matrix = csr_matrix([
        [5, 0, 3],
        [0, 4, 1]
    ], dtype=np.float32)
    
    # item embeddings: 3 items x 3 dims
    item_embeddings = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ], dtype=np.float32)
    
    return UserCollaborativeModel(index, user_item_matrix, item_embeddings)


def test_user_recommend_basic(mock_user_collaborative_model: UserCollaborativeModel):
    # Ratings: user rated item 0
    ratings = {0: 5.0}
    top_n = 2
    k_similar_users = 2
    
    res = mock_user_collaborative_model.recommend(ratings, top_n, k_similar_users)
    
    # Should return tuples
    for r in res:
        assert isinstance(r, tuple)
        assert isinstance(r[0], int)
        assert isinstance(r[1], float)
    
    # Already rated item 0 should not appear
    item_ids = [i for i, _ in res]
    assert 0 not in item_ids


def test_user_recommend_top_n_exceeds(mock_user_collaborative_model: UserCollaborativeModel):
    ratings = {0: 5.0}
    res = mock_user_collaborative_model.recommend(ratings, top_n=10, k_similar_users=2)
    # Should not exceed number of items minus already rated
    assert len(res) <= mock_user_collaborative_model.num_items - len(ratings)


def test_user_recommend_no_ratings(mock_user_collaborative_model: UserCollaborativeModel):
    # User hasn't rated anything
    ratings = {}
    res = mock_user_collaborative_model.recommend(ratings, top_n=2, k_similar_users=2)
    # Should still return top items
    assert len(res) <= mock_user_collaborative_model.num_items
