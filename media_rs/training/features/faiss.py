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
    """
    Saves FAISS index

    Args:
        index (faiss.Index): FAISS index
        path (str): Path to save
    """
    
    faiss.write_index(index, path)

def load_faiss_index(path: str) -> faiss.Index:
    """
    Load FAISS index

    Args:
        path (str): Location of index file to load

    Returns:
        faiss.Index: FAISS index of the top-K most similar users
    """
    return faiss.read_index(path)
