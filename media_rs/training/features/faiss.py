import faiss
import numpy as np
from typing import Dict, List, Tuple

from enum import Enum

class FaissMethod(str, Enum):
    COSINE = "cosine"
    L2 = "l2"

# ----------------------------
# 1. Build FAISS index
# ----------------------------
def build_faiss_index(
    embeddings: np.ndarray,
    n_dims: int = None,
    metric: FaissMethod = FaissMethod.COSINE
) -> faiss.Index:
    """
    Build a FAISS index for approximate nearest neighbors

    Args:
        embeddings (np.ndarray): 
            Embeddings of shape (num_items, dim)
        
        n_dims (int): 
            Number of dimensions
        
        metric (FaissMethod): 
            Method to use for calculating

    Raises:
        ValueError: If invalid FaissMethod

    Returns:
        faiss.Index: 
            FAISS index of the top-K most similar users

    """
    if n_dims is None:
        n_dims = embeddings.shape[1]

    if metric.value == "cosine":
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        index = faiss.IndexFlatIP(n_dims)  # Inner product = cosine for normalized vectors
    elif metric.value == "l2":
        index = faiss.IndexFlatL2(n_dims)
    else:
        raise ValueError("metric must be 'cosine' or 'l2'")

    index.add(embeddings)
    return index

# ----------------------------
# 2. Query Top-K neighbors
# ----------------------------
def query_faiss_topk(
    index: faiss.Index,
    embeddings: np.ndarray,
    k: int = 100
) -> Dict[int, List[Tuple[int, float]]]:
    """
    Query top-K neighbors for each embedding in embeddings
    Returns dictionary: item_id -> list of (neighbor_id, score)
    """
    """
    Query top-K neighbors for each embedding in embeddings

    Args: 
        index (faiss.Index): 
            FAISS index of the top-K most similar users
            
        embeddings (np.ndarray): 
            Embeddings of shape (num_items, dim)
            
        k (int):
            Number of nearest neighbours to query. Defaults to 100.

    Returns:
        Dict[int, List[Tuple[int, float]]]: 
            Dictionary of node index, and it's K nearest neighbours index
            position and distance
        
    """
    if k <= 0:
        raise ValueError("k must be a positive integer")

    if embeddings.ndim != 2:
        raise ValueError("embeddings must be a 2D array")

    num_queries = embeddings.shape[0]

    if num_queries == 0:
        return {}

    # FAISS requires float32
    embeddings = embeddings.astype(np.float32, copy=False)

    # Max possible neighbors per item (excluding self)
    max_neighbors = index.ntotal - 1
    if max_neighbors < 1:
        raise ValueError("FAISS index must contain at least 2 vectors")

    # Clamp k safely
    k = min(k, max_neighbors)

    # Search k+1 so we can drop self-matches
    distances, indices = index.search(embeddings, k + 1)

    topk_dict: Dict[int, List[Tuple[int, float]]] = {}

    for i in range(num_queries):
        neighbors = [
            (int(n), float(d))
            for n, d in zip(indices[i], distances[i])
            if n != i
        ][:k]

        topk_dict[i] = neighbors

    return topk_dict