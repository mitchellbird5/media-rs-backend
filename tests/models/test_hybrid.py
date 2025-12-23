import pytest
import numpy as np
from src.models.hybrid import HybridModel
from src.models.content import ContentModel
from src.models.collab import CollaborativeModel
from src.types.model import CollabMethod


@pytest.fixture
def sample_content_model():
    ids = [1, 2, 3]
    features = [
        "apple banana",
        "banana orange",
        "apple orange banana"
    ]
    # k=2 for small test
    return ContentModel(ids=ids, features=features, k=2)


@pytest.fixture
def sample_collaborative_model():
    ids = [1, 2, 3]
    # Small user-item matrix (items x users)
    user_item_matrix = np.array([
        [5, 3, 0],  # item 1
        [4, 0, 0],  # item 2
        [1, 1, 0],  # item 3
    ])
    return CollaborativeModel(
        ids=ids,
        user_item_matrix=user_item_matrix,
        collab_method=CollabMethod.ITEM,
        k=2
    )


@pytest.fixture
def hybrid_model(sample_content_model, sample_collaborative_model):
    ids = [1, 2, 3]
    return HybridModel(
        ids=ids,
        content_model=sample_content_model,
        collaborative_model=sample_collaborative_model,
        alpha=0.6
    )


def test_initialization(hybrid_model, sample_content_model, sample_collaborative_model):
    assert hybrid_model.ids == [1, 2, 3]
    assert hybrid_model.content_model is sample_content_model
    assert hybrid_model.collaborative_model is sample_collaborative_model
    assert hybrid_model.alpha == 0.6


def test_recommend_returns_top_n(hybrid_model):
    top_n = 2
    recs = hybrid_model.recommend(1, top_n=top_n)

    assert isinstance(recs, list)
    assert len(recs) <= top_n
    assert all(isinstance(x, int) for x in recs)


def test_alpha_blending_effect(sample_content_model, sample_collaborative_model):
    hybrid = HybridModel(
        ids=[1, 2, 3],
        content_model=sample_content_model,
        collaborative_model=sample_collaborative_model,
        alpha=1.0
    )
    recs_alpha_1 = hybrid.recommend(1, top_n=3)

    hybrid.alpha = 0.0
    recs_alpha_0 = hybrid.recommend(1, top_n=3)

    # Instead of asserting full lists differ, assert the **score of top item is from correct model**
    top_content_score_id = max(
        dict(sample_content_model.neighbors[1]), key=lambda x: sample_content_model.neighbors[1][[n[0] for n in sample_content_model.neighbors[1]].index(x)][1]
    )
    top_collab_score_id = max(
        dict(sample_collaborative_model.neighbors[1]), key=lambda x: sample_collaborative_model.neighbors[1][[n[0] for n in sample_collaborative_model.neighbors[1]].index(x)][1]
    )

    # Top recommendation with alpha=1.0 should match content model top
    assert recs_alpha_1[0] == top_content_score_id
    # Top recommendation with alpha=0.0 should match collaborative model top
    assert recs_alpha_0[0] == top_collab_score_id


def test_recommend_top_n_larger_than_available(hybrid_model):
    recs = hybrid_model.recommend(1, top_n=10)
    # Should return at most all available items
    assert len(recs) <= len(hybrid_model.ids)


def test_recommend_consistency(hybrid_model):
    # Multiple calls return the same result
    rec1 = hybrid_model.recommend(1, top_n=3)
    rec2 = hybrid_model.recommend(1, top_n=3)
    assert rec1 == rec2


def test_recommend_raises_for_unknown_id(hybrid_model):
    with pytest.raises(KeyError):
        # HybridModel accesses neighbors dict directly
        hybrid_model.recommend(999, top_n=2)
