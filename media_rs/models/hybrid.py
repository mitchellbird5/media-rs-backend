# src/models/hybrid.py
import pandas as pd

from typing import List
from rs_types.model import IdType

from models.base import BaseRecommender
from models.content import ContentModel
from models.collab import CollaborativeModel

class HybridModel(BaseRecommender):
    def __init__(
        self,
        ids: List[IdType],
        content_model: ContentModel,
        collaborative_model: CollaborativeModel,
        alpha: float = 0.7
    ):
        self.ids = ids
        self.content_model = content_model
        self.collaborative_model = collaborative_model
        self.alpha = alpha

    def recommend(self, item_id: IdType, top_n: int) -> List[int]:
        content = dict(self.content_model.neighbors[item_id])
        item = dict(self.collaborative_model.neighbors[item_id])

        all_ids = set(content) | set(item)

        scores = {
            mid: self.alpha * content.get(mid, 0.0)
            + (1 - self.alpha) * item.get(mid, 0.0)
            for mid in all_ids
        }

        return (
            pd.Series(scores)
            .sort_values(ascending=False)
            .head(top_n)
            .index.tolist()
        )
