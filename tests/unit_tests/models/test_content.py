# tests/unit_tests/models/test_content.py
import pytest
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

from media_rs.serving.recommender.models.content import (
    ContentSimilaritySBERTModel,
    ContentSimilarityTFIDFModel,
)

# Define IdType and ContentSimilarity for testing purposes
IdType = int
ContentSimilarity = tuple[int, float]


# --------------------------
# Fixtures
# --------------------------

@pytest.fixture
def mock_topk_graph():
    return {
        0: [(1, 0.9), (2, 0.8), (3, 0.7)],
        1: [(0, 0.9), (2, 0.85)],
        2: [(0, 0.8), (1, 0.85)],
        3: [(0, 0.7)],
    }


@pytest.fixture
def mock_sbert_transformer():
    # Dummy transformer for testing
    class MockTransformer:
        def encode(self, texts, convert_to_numpy=True, normalize_embeddings=True):
            # Return fixed embedding of same dimension as mock_embeddings
            return np.array([[0.1, 0.2, 0.3, 0.4, 0.5]], dtype=np.float32)
    return MockTransformer()


@pytest.fixture
def mock_sbert_embeddings():
    # 4 items, embedding_dim = 5
    return np.array([
        [0.1, 0.2, 0.3, 0.4, 0.5],
        [0.5, 0.4, 0.3, 0.2, 0.1],
        [0.2, 0.1, 0.4, 0.3, 0.5],
        [0.3, 0.2, 0.1, 0.5, 0.4],
    ], dtype=np.float32)


@pytest.fixture
def mock_tfidf_components_and_embeddings():
    # 4 items in corpus
    corpus = ["toy story", "forrest gump", "chronicles of narnia", "wizard of oz"]
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(corpus)

    svd = TruncatedSVD(n_components=5, random_state=42)
    embeddings = svd.fit_transform(tfidf_matrix).astype(np.float32)

    # Normalize embeddings
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / np.where(norms == 0, 1, norms)

    return vectorizer, svd, embeddings


# --------------------------
# Tests
# --------------------------

@pytest.mark.parametrize("model_type", ["SBERT", "TFIDF"])
def test_recommend_and_from_description(
    mock_topk_graph,
    mock_sbert_transformer,
    mock_sbert_embeddings,
    mock_tfidf_components_and_embeddings,
    model_type
):
    topk_graph = mock_topk_graph
    top_n = 2
    description = "fun movie about adventure"

    # Instantiate model
    if model_type == "SBERT":
        model = ContentSimilaritySBERTModel(
            topk_graph=topk_graph,
            embeddings=mock_sbert_embeddings,
            transformer=mock_sbert_transformer
        )
    else:
        vectorizer, svd, embeddings = mock_tfidf_components_and_embeddings
        model = ContentSimilarityTFIDFModel(
            topk_graph=topk_graph,
            embeddings=embeddings,  # <— correct embeddings
            vectorizer=vectorizer,
            svd=svd
        )

    # --- Test recommend by item_id ---
    result = model.recommend(0, top_n)
    assert len(result) == top_n
    for r in result:
        assert isinstance(r, tuple)
        assert isinstance(r[0], int)
        assert isinstance(r[1], float)

    # Top-N exceeding available neighbors
    result = model.recommend(1, 10)
    assert len(result) == len(topk_graph[1])

    # --- Test recommend from description ---
    desc_result = model.recommend_from_description(description, top_n)
    assert len(desc_result) == top_n
    for r in desc_result:
        assert isinstance(r, tuple)
        assert isinstance(r[0], int)
        assert isinstance(r[1], float)

    # Check descending order of similarity scores
    scores = [score for _, score in desc_result]
    assert all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1))


def test_recommend_from_description_top_n_exceeds(
    mock_topk_graph,
    mock_sbert_transformer,
    mock_sbert_embeddings,
    mock_tfidf_components_and_embeddings
):
    top_n = 10
    description = "adventure movie"

    # SBERT
    sbert_model = ContentSimilaritySBERTModel(
        topk_graph=mock_topk_graph,
        embeddings=mock_sbert_embeddings,
        transformer=mock_sbert_transformer
    )
    result = sbert_model.recommend_from_description(description, top_n)
    # Should return at most number of items
    assert len(result) == mock_sbert_embeddings.shape[0]

    # TFIDF
    vectorizer, svd, embeddings = mock_tfidf_components_and_embeddings
    tfidf_model = ContentSimilarityTFIDFModel(
        topk_graph=mock_topk_graph,
        embeddings=embeddings,  # <— correct embeddings
        vectorizer=vectorizer,
        svd=svd
    )
    result = tfidf_model.recommend_from_description(description, top_n)
    assert len(result) == embeddings.shape[0]
