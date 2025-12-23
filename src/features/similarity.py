import numpy as np
from sklearn.neighbors import NearestNeighbors
from typing import List, Dict, Any

from src.types.model import ContentSimilarity


def compute_topk_similarity(
    matrix: np.ndarray,
    index_labels: List[Any] | None = None,
    k: int = 50
) -> Dict[Any, List[ContentSimilarity]]:
    """
    Compute top-k nearest neighbors using cosine similarity.
    Returns dict {item_id: [(neighbor_id, similarity), ...]}
    """
    n_samples = matrix.shape[0]
    k = min(k, n_samples - 1)
    
    model = NearestNeighbors(metric="cosine", n_neighbors=k+1)
    model.fit(matrix)
    distances, indices = model.kneighbors(matrix)

    results: Dict[Any, List[ContentSimilarity]] = {}

    for row_idx, (dists, nbrs) in enumerate(zip(distances, indices)):
        key = index_labels[row_idx] if index_labels is not None else row_idx
        neighbors = []

        for dist, nbr_idx in zip(dists[1:], nbrs[1:]):  # skip self
            sim = 1 - dist
            nbr_key = index_labels[nbr_idx] if index_labels is not None else nbr_idx
            neighbors.append((nbr_key, sim))
        results[key] = neighbors

    return results
