# tests/test_hybrid_model.py
import pytest
import numpy as np
from typing import Dict, List, Tuple

from media_rs.serving.recommender.models.hybrid import HybridModel
from media_rs.rs_types.model import ContentSimilarity

# --- Mock models ---
class MockContentModel:
    def recommend(self, item_idx: int, top_n: int) -> List[ContentSimilarity]:
        return [(0, 0.9), (1, 0.8), (2, 0.7)][:top_n]

class MockItemCollabModel:
    def recommend(self, item_idx: int, top_n: int) -> List[ContentSimilarity]:
        return [(1, 0.85), (2, 0.75), (3, 0.65)][:top_n]

class MockUserCollabModel:
    def recommend(self, ratings: Dict[int, float], top_n: int, k_similar_users: int) -> List[ContentSimilarity]:
        return [(2, 0.95), (3, 0.6), (4, 0.5)][:top_n]


# --- Fixtures ---
@pytest.fixture
def hybrid_model():
    content_model = MockContentModel()
    item_collab_model = MockItemCollabModel()
    user_collab_model = MockUserCollabModel()
    return HybridModel(
        content_model=content_model,
        item_collab_model=item_collab_model,
        user_collab_model=user_collab_model,
        alpha=0.4,
        beta=0.3
    )


# --- Tests ---
def test_combine_scores(hybrid_model: HybridModel):
    content_scores = {0: 0.9, 1: 0.8, 2: 0.7}
    item_scores = {1: 0.85, 2: 0.75, 3: 0.65}
    user_scores = {2: 0.95, 3: 0.6, 4: 0.5}

    combined = hybrid_model._combine_scores(content_scores, item_scores, user_scores)

    # Check all keys are included
    assert set(combined.keys()) == {0, 1, 2, 3, 4}

    # Check weighted sum calculation
    gamma = 1.0 - hybrid_model.alpha - hybrid_model.beta
    assert combined[2] == pytest.approx(0.4*0.7 + 0.3*0.75 + gamma*0.95)
    assert combined[0] == pytest.approx(0.4*0.9 + 0.3*0 + gamma*0)
    assert combined[4] == pytest.approx(0.4*0 + 0.3*0 + gamma*0.5)


def test_top_n(hybrid_model: HybridModel):
    scores = {0: 0.9, 1: 0.8, 2: 0.95, 3: 0.6}
    top_items = hybrid_model._top_n(scores, top_n=2)

    # Should return 2 items
    assert len(top_items) == 2

    # Check items are sorted descending
    scores_list = [score for _, score in top_items]
    assert scores_list[0] >= scores_list[1]


def test_recommend(hybrid_model: HybridModel):
    ratings = {0: 5.0, 1: 3.0}
    top_n = 3
    k_similar_users = 2

    results = hybrid_model.recommend(
        item_idx=0,
        ratings=ratings,
        k_similar_users=k_similar_users,
        top_n=top_n
    )

    # Should return top_n items
    assert len(results) == top_n

    # Each element is (int, float)
    for r in results:
        assert isinstance(r, tuple)
        assert isinstance(r[0], int)
        assert isinstance(r[1], float)

    # Scores are sorted descending
    scores = [s for _, s in results]
    assert all(scores[i] >= scores[i+1] for i in range(len(scores)-1))
