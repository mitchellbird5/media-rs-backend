# src/models/collaborative.py
import numpy as np

from typing import List

from src.models.base import BaseRecommender
from src.features.similarity import compute_topk_similarity

from src.types.model import IdType, ContentSimilarity, CollabMethod

class CollaborativeModel(BaseRecommender):
    def __init__(
        self,
        ids: List[IdType],
        user_item_matrix: np.ndarray,
        collab_method: CollabMethod,
        k: int = 50,
    ):
        self.ids = ids
        self.k = k
        self.user_item_matrix = user_item_matrix
        self.collab_method = collab_method
        
        self._neighbors: dict | None = None

    @property
    def neighbors(self):
        if self._neighbors is None:
            self._neighbors = compute_topk_similarity(
                self._get_uim(),
                index_labels=self.ids,
                k=self.k
            )
        return self._neighbors

    def recommend(self, item_id: IdType, top_n: int) -> List[ContentSimilarity]:
        if item_id not in self.neighbors:
            raise ValueError(f"Item ID {item_id} not found")
        return self.neighbors[item_id][:top_n]

    def _get_uim(self):
        if self.collab_method==CollabMethod.ITEM:
            return self.user_item_matrix
        elif self.collab_method==CollabMethod.USER:
            return self.user_item_matrix.T
        else:
            raise ValueError(f"Incompatible method {self.collab_method}")