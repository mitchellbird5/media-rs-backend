# src/models/hybrid.py
import pandas as pd
from typing import List

from media_rs.models.base import BaseRecommender
from media_rs.models.content import ContentModel
from media_rs.models.collab import ItemItemCollaborativeModel, UserCollaborativeModel

import pandas as pd
from typing import List

class HybridModel(BaseRecommender):
    def __init__(
        self,
        content_model: ContentModel,        # ContentModel instance
        item_collab_model: ItemItemCollaborativeModel,    # ItemItemCollaborativeModel instance
        user_collab_model: UserCollaborativeModel,    # UserCollaborativeModel instance
        alpha: float = 0.5,   # weight for content
        beta: float = 0.3     # weight for item CF (user CF gets 1-alpha-beta)
    ):
        self.content_model = content_model
        self.item_collab_model = item_collab_model
        self.user_collab_model = user_collab_model
        self.alpha = alpha
        self.beta = beta
        self.gamma = 1.0 - alpha - beta

    def recommend(self, item_idx: int, user_idx: int, top_n: int = 10) -> List[int]:
        # 1. Content scores
        content_scores = dict(self.content_model.recommend(item_idx, top_n=100))

        # 2. Item-item CF scores
        item_scores = dict(self.item_collab_model.recommend(item_idx, top_n=100))

        # 3. User-user CF scores
        user_scores = dict(self.user_collab_model.recommend(user_idx, top_n=100))

        # 4. Combine scores
        all_ids = set(content_scores) | set(item_scores) | set(user_scores)
        combined_scores = {
            i: self.alpha * content_scores.get(i, 0.0) +
               self.beta  * item_scores.get(i, 0.0) +
               self.gamma * user_scores.get(i, 0.0)
            for i in all_ids
        }

        # 5. Sort descending and return top-N indices
        top_indices = pd.Series(combined_scores).sort_values(ascending=False).head(top_n).index.tolist()
        return [(i, float(combined_scores[i])) for i in top_indices]
