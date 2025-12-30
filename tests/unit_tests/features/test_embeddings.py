# tests/test_embedding_utils.py
import pytest
import numpy as np
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
    mock_model.encode.assert_called_once_with(texts, convert_to_numpy=True, normalize_embeddings=True)


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
    item_embeddings = np.array([[1, 0], [0, 1], [1, 1]])
    user_item_dict = {0: [0, 2], 1: [1]}

    user_embeds = compute_user_embeddings(user_item_dict, item_embeddings)

    assert isinstance(user_embeds, dict)
    assert len(user_embeds) == 2

    # Check computed embeddings
    np.testing.assert_array_almost_equal(user_embeds[0], np.mean(item_embeddings[[0, 2]], axis=0))
    np.testing.assert_array_almost_equal(user_embeds[1], item_embeddings[1])

