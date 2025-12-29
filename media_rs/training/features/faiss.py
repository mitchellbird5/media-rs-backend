import faiss
import numpy as np
from typing import Dict, List, Tuple

# ----------------------------
# 1. Build FAISS index
# ----------------------------
def build_faiss_index(
    embeddings: np.ndarray,
    n_dims: int = None,
    metric: str = "cosine"
) -> faiss.Index:
    """
    Build a FAISS index for approximate nearest neighbors
    embeddings: np.ndarray of shape (num_items, dim)
    metric: 'cosine' or 'l2'
    """
    if n_dims is None:
        n_dims = embeddings.shape[1]

    if metric == "cosine":
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        index = faiss.IndexFlatIP(n_dims)  # Inner product = cosine for normalized vectors
    elif metric == "l2":
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
    D, I = index.search(embeddings, k)  # D = distances, I = indices
    topk_dict = {}
    for i, (indices, distances) in enumerate(zip(I, D)):
        # For cosine similarity, distances = inner product
        topk_dict[i] = list(zip(indices.tolist(), distances.tolist()))
    return topk_dict

# ----------------------------
# 3. Save / Load FAISS index
# ----------------------------
def save_faiss_index(index: faiss.Index, path: str):
    faiss.write_index(index, path)

def load_faiss_index(path: str) -> faiss.Index:
    return faiss.read_index(path)
