# tests/test_embedding_utils.py
import pytest
import numpy as np
from scipy.sparse import csr_matrix
import faiss
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sentence_transformers import SentenceTransformer
from unittest.mock import MagicMock, patch

from media_rs.training.features.embeddings import (
    compute_sbert_embeddings,
    compute_tfidf_embeddings,
    compute_user_embeddings
)


# -----------------------------
# Test compute_sbert_embeddings
# -----------------------------
@patch("media_rs.training.features.embeddings.SentenceTransformer")
def test_compute_sbert_embeddings(mock_sbert):
    # Mock encode method
    mock_model = MagicMock()
    mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    mock_sbert.return_value = mock_model

    texts = ["Hello world", "Test sentence"]
    model, embeddings = compute_sbert_embeddings(texts)

    # Ensure the model returned is the mock
    assert model == mock_model
    # Ensure embeddings shape matches input
    assert embeddings.shape == (2, 3)
    # Ensure encode was called
    mock_model.encode.assert_called_once_with(texts, convert_to_numpy=True)


# -----------------------------
# Test compute_tfidf_embeddings
# -----------------------------
def test_compute_tfidf_embeddings_default():
    texts = ["Hello world", "Test sentence", "Another text"]
    embeddings, vectorizer, svd = compute_tfidf_embeddings(texts, n_features=10, n_components=2)

    # Check types
    assert isinstance(embeddings, np.ndarray)
    assert isinstance(vectorizer, TfidfVectorizer)
    assert isinstance(svd, TruncatedSVD)

    # Check shapes
    assert embeddings.shape == (3, 2)

    # Check that TFIDF vectorizer learned vocabulary
    assert hasattr(vectorizer, "vocabulary_")
    assert len(vectorizer.vocabulary_) <= 10


def test_compute_tfidf_embeddings_with_existing_vectorizer_svd():
    texts = ["Hello world", "Test sentence"]
    vectorizer = TfidfVectorizer(max_features=5).fit(texts)
    svd = TruncatedSVD(n_components=1, random_state=42).fit(vectorizer.transform(texts))

    embeddings, v, s = compute_tfidf_embeddings(texts, vectorizer=vectorizer, svd=svd)
    
    # Should reuse the same objects
    assert v == vectorizer
    assert s == svd
    # Shape should match n_components
    assert embeddings.shape == (2, 1)


# -----------------------------
# Test compute_user_embeddings
# -----------------------------
def test_compute_user_embeddings():
    # Item embeddings: 3 items, 2 dims
    item_embeddings = np.array(
        [[1.0, 0.0],
         [0.0, 1.0],
         [1.0, 1.0]],
        dtype=np.float32
    )

    # User–item ratings matrix
    # user 0 rated items 0 and 2
    # user 1 rated item 1
    data = np.array([1.0, 3.0, 2.0], dtype=np.float32)
    rows = np.array([0, 0, 1])
    cols = np.array([0, 2, 1])

    user_item_matrix = csr_matrix((data, (rows, cols)), shape=(2, 3))

    user_embeds = compute_user_embeddings(user_item_matrix, item_embeddings)

    # Basic shape/type checks
    assert isinstance(user_embeds, np.ndarray)
    assert user_embeds.shape == (2, 2)

    # ---- Expected embeddings (before normalization) ----
    # User 0:
    # ratings = [1, 3] → mean = 2 → centered = [-1, 1]
    v0 = np.mean(
        item_embeddings[[0, 2]] * np.array([-1.0, 1.0])[:, None],
        axis=0,
    )

    # User 1:
    # ratings = [2] → mean = 2 → centered = [0]
    v1 = np.array([0.0, 0.0], dtype=np.float32)

    expected = np.vstack([v0, v1]).astype(np.float32)
    faiss.normalize_L2(expected)

    np.testing.assert_array_almost_equal(user_embeds, expected)
