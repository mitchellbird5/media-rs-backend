# tests/test_content_similarity_model.py
import pytest
import numpy as np
from sentence_transformers import SentenceTransformer

from media_rs.serving.recommender.models.content import ContentSimilarityModel

# Define IdType and ContentSimilarity for testing purposes
IdType = int
ContentSimilarity = tuple[int, float]


@pytest.fixture
def mock_model():
    # Mock topk_graph
    topk_graph = {
        0: [(1, 0.9), (2, 0.8), (3, 0.7)],
        1: [(0, 0.9), (2, 0.85)]
    }

    # Mock embeddings (3 items, 5-dimensional)
    embeddings = np.array([
        [0.1, 0.2, 0.3, 0.4, 0.5],
        [0.5, 0.4, 0.3, 0.2, 0.1],
        [0.2, 0.1, 0.4, 0.3, 0.5]
    ], dtype="float32")

    # Mock SentenceTransformer
    class MockTransformer:
        def encode(self, texts, convert_to_numpy=True, normalize_embeddings=True):
            # return deterministic small embeddings for testing
            return np.array([[0.1, 0.2, 0.3, 0.4, 0.5]])

    transformer = MockTransformer()

    return ContentSimilarityModel(topk_graph, embeddings, transformer)


def test_recommend_top_n(mock_model):
    # Test normal usage
    result = mock_model.recommend(0, 2)
    assert len(result) == 2
    assert result == [(1, 0.9), (2, 0.8)]

    # Test top_n larger than available
    result = mock_model.recommend(1, 5)
    assert len(result) == 2
    assert result == [(0, 0.9), (2, 0.85)]


def test_recommend_from_description(mock_model):
    # Use description to get recommendations
    description = "Some test description"
    result = mock_model.recommend_from_description(description, top_n=2)

    # Should return 2 items
    assert len(result) == 2

    # Check structure: list of tuples (int, float)
    for r in result:
        assert isinstance(r, tuple)
        assert isinstance(r[0], int)
        assert isinstance(r[1], float)

    # Check that scores are in descending order
    scores = [score for _, score in result]
    assert all(scores[i] >= scores[i+1] for i in range(len(scores)-1))


def test_recommend_from_description_top_n_exceeds(mock_model):
    result = mock_model.recommend_from_description("desc", top_n=10)
    # Should not exceed number of embeddings
    assert len(result) == mock_model.embeddings.shape[0]
