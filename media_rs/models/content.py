# src/models/content.py
import numpy as np

from typing import List, Dict

from features.similarity import compute_topk_similarity
from features.tfidf import compute_tfidf_matrix
from models.base import BaseRecommender

from rs_types.model import IdType, ContentSimilarity

class ContentModel(BaseRecommender):
    """
    Generic content-based recommender.

    Items can be any type as long as they are represented as dicts:
        {
            "id": unique identifier,
            "features": string (concatenated features)
        }
    """
    def __init__(
        self,
        ids: List[IdType],
        features: List[str],
        k: int = 50,
        max_features: int = 5000
    ):
        self.ids = ids
        self.features = features
        self.k = k
        self.max_features = max_features

        self._tfidf_matrix: np.ndarray | None = None
        self._neighbors: dict | None = None

    @property
    def tfidf_matrix(self) -> np.ndarray:
        if self._tfidf_matrix is None:
            self._tfidf_matrix = compute_tfidf_matrix(self.features)
        return self._tfidf_matrix

    @property
    def neighbors(self) -> Dict[IdType, List[ContentSimilarity]]:
        if self._neighbors is None:
            self._neighbors = compute_topk_similarity(
                self.tfidf_matrix,
                index_labels=self.ids,
                k=self.k
            )
        return self._neighbors

    def recommend(self, item_id: IdType, top_n: int) -> List[ContentSimilarity]:
        if item_id not in self.neighbors:
            raise ValueError(f"Item ID {item_id} not found")
        return self.neighbors[item_id][:top_n]

