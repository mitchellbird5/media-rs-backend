# tests/test_topk_builders.py
import numpy as np
import pytest
from scipy.sparse import csr_matrix
import faiss

from media_rs.training.build.build_topk_graphs import (
    build_item_cf_topk,
    build_topk_content
)

# -----------------------------
# Fixtures
# -----------------------------
@pytest.fixture
def user_item_matrix():
    # 3 users x 4 items
    data = np.array([
        [5, 0, 3, 1],
        [0, 4, 1, 0],
        [2, 0, 0, 3]
    ], dtype=np.float32)
    return csr_matrix(data)

@pytest.fixture
def item_embeddings():
    # 4 items, 3-dim embeddings
    np.random.seed(42)
    return np.random.rand(4, 3).astype(np.float32)

# -----------------------------
# Test build_item_cf_topk
# -----------------------------
def test_build_item_cf_topk_basic(user_item_matrix):
    topk_cf = build_item_cf_topk(user_item_matrix, k=2)
    
    assert isinstance(topk_cf, dict)
    # Should have one entry per item
    assert len(topk_cf) == user_item_matrix.shape[1]
    
    for neighbors in topk_cf.values():
        # Each neighbor list should have <= k entries
        assert len(neighbors) <= 2
        for item_idx, score in neighbors:
            assert isinstance(item_idx, int)
            assert isinstance(score, float)

def test_build_item_cf_topk_k_greater_than_items(user_item_matrix):
    # k greater than number of items
    topk_cf = build_item_cf_topk(user_item_matrix, k=10)
    for neighbors in topk_cf.values():
        # Should not exceed total items - 1 (exclude self)
        assert len(neighbors) <= user_item_matrix.shape[1] - 1

def test_build_item_cf_topk_batch_size(user_item_matrix):
    # Batch size smaller than number of items
    topk_cf = build_item_cf_topk(user_item_matrix, k=2, batch_size=2)
    # Check all items present
    assert set(topk_cf.keys()) == set(range(user_item_matrix.shape[1]))

# -----------------------------
# Test build_topk_content
# -----------------------------
def test_build_topk_content_basic(item_embeddings):
    topk_content = build_topk_content(item_embeddings, top_k=2)
    
    assert isinstance(topk_content, dict)
    assert len(topk_content) == item_embeddings.shape[0]
    
    for neighbors in topk_content.values():
        # Each neighbor list should have <= top_k entries
        assert len(neighbors) <= 2
        for item_idx, score in neighbors:
            assert isinstance(item_idx, int)
            assert isinstance(score, float)

def test_build_topk_content_top_k_greater_than_items(item_embeddings):
    topk_content = build_topk_content(item_embeddings, top_k=10)
    for neighbors in topk_content.values():
        # Should not exceed total items - 1 (exclude self)
        assert len(neighbors) <= item_embeddings.shape[0] - 1

def test_build_topk_content_self_excluded(item_embeddings):
    topk_content = build_topk_content(item_embeddings, top_k=3)
    for idx, neighbors in topk_content.items():
        neighbor_indices = [n for n, _ in neighbors]
        # Self should not be included
        assert idx not in neighbor_indices
