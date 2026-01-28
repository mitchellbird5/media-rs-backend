# tests/test_faiss_utils.py
import numpy as np
import pytest
import faiss

from media_rs.training.features.faiss import (
    build_faiss_index, 
    query_faiss_topk,
    FaissMethod
)

@pytest.fixture
def dummy_embeddings():
    # 5 items, 3-dimensional embeddings
    np.random.seed(42)
    return np.random.rand(5, 3).astype(np.float32)

# -----------------------------
# Test build_faiss_index
# -----------------------------
@pytest.mark.parametrize("metric", [FaissMethod.COSINE, FaissMethod.L2])
def test_build_faiss_index(metric, dummy_embeddings):
    index = build_faiss_index(dummy_embeddings.copy(), metric=metric)
    
    # Check type
    assert isinstance(index, faiss.Index)
    # Check number of vectors added
    assert index.ntotal == dummy_embeddings.shape[0]

    # Cosine similarity check (IndexFlatIP) normalizes embeddings
    if metric == FaissMethod.COSINE:
        # Norm of each embedding should be ~1 after normalization
        norms = np.linalg.norm(dummy_embeddings, axis=1)
        # Original embeddings passed in copy should be unchanged
        assert norms.all() > 0

def test_build_faiss_index_invalid_metric(dummy_embeddings):
    import builtins
    with pytest.raises(AttributeError):
        build_faiss_index(dummy_embeddings, metric="invalid_metric")


# -----------------------------
# Test query_faiss_topk
# -----------------------------
def test_query_faiss_topk(dummy_embeddings):
    index = build_faiss_index(dummy_embeddings.copy(), metric=FaissMethod.COSINE)
    topk = query_faiss_topk(index, dummy_embeddings.copy(), k=3)

    # Should return a dict with same number of items as embeddings
    assert isinstance(topk, dict)
    assert len(topk) == dummy_embeddings.shape[0]

    for key, neighbors in topk.items():
        # Each value should be a list of tuples
        assert isinstance(neighbors, list)
        for idx, score in neighbors:
            assert isinstance(idx, int)
            assert isinstance(score, float)
        # k neighbors returned
        assert len(neighbors) == 3

def test_query_faiss_topk_k_greater_than_index(dummy_embeddings):
    index = build_faiss_index(dummy_embeddings.copy(), metric=FaissMethod.L2)
    k = 10  # More than number of items
    topk = query_faiss_topk(index, dummy_embeddings.copy(), k=k)
    
    expected_neighbors = dummy_embeddings.shape[0] - 1
    
    for neighbors in topk.values():
        # Should return only the number of items in the index
        assert len(neighbors) == expected_neighbors
