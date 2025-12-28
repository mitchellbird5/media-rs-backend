# src/models/hybrid.py
import pandas as pd
import numpy as np

from typing import List, Dict
from media_rs.rs_types.model import ContentSimilarity

from media_rs.models.content import ContentModel
from media_rs.models.collab import ItemItemCollaborativeModel, UserCollaborativeModel

class HybridModel:
    def __init__(
        self,
        content_model: ContentModel,
        item_collab_model: ItemItemCollaborativeModel,
        user_collab_model: UserCollaborativeModel,
        alpha: float = 0.5,   # weight for content
        beta: float = 0.3     # weight for item CF
    ):
        self.content_model = content_model
        self.item_collab_model = item_collab_model
        self.user_collab_model = user_collab_model
        self.alpha = alpha
        self.beta = beta
        self.gamma = 1.0 - alpha - beta

    def recommend_existing_user(
        self, 
        item_idx: int, 
        user_idx: int, 
        top_n: int = 10
    ) -> List[ContentSimilarity]:
        """
        Recommend for an existing user (by user_idx)
        """
        # 1. Content scores
        content_scores = dict(self.content_model.recommend(item_idx, top_n=100))

        # 2. Item-item CF scores
        item_scores = dict(self.item_collab_model.recommend(item_idx, top_n=100))

        # 3. User-user CF scores
        user_scores = dict(self.user_collab_model.recommend_existing_user(user_idx, top_n=100))

        # 4. Combine scores
        combined_scores = self._combine_scores(content_scores, item_scores, user_scores)

        # 5. Return top-N
        return self._top_n(combined_scores, top_n)

    def recommend_from_ratings(
        self,
        item_idx: int,
        new_user_ratings: Dict[int, float],
        item_embeddings: np.ndarray,
        top_n: int = 10
    ) -> List[ContentSimilarity]:
        """
        Recommend for a new user given a ratings dict {item_idx: rating}
        """
        # 1. Content scores
        content_scores = dict(self.content_model.recommend(item_idx, top_n=100))

        # 2. Item-item CF scores
        item_scores = dict(self.item_collab_model.recommend(item_idx, top_n=100))

        # 3. User-user CF scores using new user ratings
        user_scores = dict(self.user_collab_model.recommend_from_ratings(
            new_user_ratings, 
            item_embeddings, 
            top_n=100
        ))

        # 4. Combine scores
        combined_scores = self._combine_scores(content_scores, item_scores, user_scores)

        # 5. Return top-N
        return self._top_n(combined_scores, top_n)

    def _combine_scores(
        self,
        content_scores: Dict[int, float],
        item_scores: Dict[int, float],
        user_scores: Dict[int, float]
    ) -> Dict[int, float]:
        """
        Combine scores with weighted sum
        """
        all_ids = set(content_scores) | set(item_scores) | set(user_scores)
        return {
            i: self.alpha * content_scores.get(i, 0.0) +
               self.beta  * item_scores.get(i, 0.0) +
               self.gamma * user_scores.get(i, 0.0)
            for i in all_ids
        }

    def _top_n(self, scores: Dict[int, float], top_n: int) -> List[ContentSimilarity]:
        """
        Sort and return top-N items as (item_idx, score)
        """
        sorted_items = pd.Series(scores).sort_values(ascending=False).head(top_n)
        return [(i, float(sorted_items[i])) for i in sorted_items.index]