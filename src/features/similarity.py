# utils/similarity.py
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from typing import List, Dict, Any

from src.types.model import ContentSimilarity

def compute_tfidf_matrix(combined_text: List[str], max_features=5000):
    """
    Returns the TF-IDF matrix.
    """
    vectorizer = TfidfVectorizer(stop_words='english', max_features=max_features)
    tfidf_matrix = vectorizer.fit_transform(combined_text)
    return tfidf_matrix

def compute_topk_similarity(
    matrix: np.ndarray,
    index_labels: List[Any] | None = None,
    k: int = 50
) -> Dict[Any, List[ContentSimilarity]]:
    """
    Compute top-k nearest neighbors using cosine similarity.
    Returns dict {item_id: [(neighbor_id, similarity), ...]}
    """
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
