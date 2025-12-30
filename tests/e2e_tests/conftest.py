# tests/conftest.py
import pytest
import numpy as np
from rest_framework.test import APIClient
from scipy.sparse import csr_matrix
import faiss
from sentence_transformers import SentenceTransformer

from media_rs.serving.recommender.models.content import ContentSimilarityModel
from media_rs.serving.recommender.models.collab import ItemItemCollaborativeModel, UserCollaborativeModel

# -----------------------------
# Content model fixture
# -----------------------------
@pytest.fixture
def content_model():
    # Tiny item dataset
    item_texts = ["Toy Story", "Inception", "Matrix"]
    vectorizer = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = vectorizer.encode(item_texts, convert_to_numpy=True, normalize_embeddings=True)

    # Top-k graph
    topk_graph = {
        0: [(1, 0.9), (2, 0.8)],
        1: [(0, 0.9), (2, 0.7)],
        2: [(0, 0.8), (1, 0.7)],
    }

    return ContentSimilarityModel(topk_graph=topk_graph, embeddings=embeddings, transformer=vectorizer)


# -----------------------------
# Item-item collaborative model fixture
# -----------------------------
@pytest.fixture
def item_collab_model():
    topk_graph = {
        0: [(1, 0.9), (2, 0.8)],
        1: [(0, 0.9), (2, 0.7)],
        2: [(0, 0.8), (1, 0.7)],
    }
    return ItemItemCollaborativeModel(topk_graph)


# -----------------------------
# User-user collaborative model fixture
# -----------------------------
@pytest.fixture
def user_collab_model():
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

    # FAISS index over user embeddings
    user_embeddings = np.array([
        [1, 0, 0],
        [0, 1, 0]
    ], dtype=np.float32)
    faiss.normalize_L2(user_embeddings)
    index = faiss.IndexFlatIP(3)
    index.add(user_embeddings)

    return UserCollaborativeModel(faiss_index=index, user_item_matrix=user_item_matrix, item_embeddings=item_embeddings)


# -----------------------------
# API client fixture
# -----------------------------
@pytest.fixture
def api_client():
    return APIClient()
