# src/recommender.py
from typing import List, Optional
from src.types.model import IdType

from src.models.content import ContentModel
# from src.models.item_collab import ItemCollaborativeModel
# from src.models.user_collab import UserCollaborativeModel
# from src.models.hybrid import HybridModel

class RecommenderSystem:
    def __init__(
        self,
        ids: List[IdType],
        features: List[str],
        k: Optional[int] = 50,
        max_features: Optional[int] = 5000,
    ):
        self.content = ContentModel(ids, features, k, max_features)
        # self.collaborative_item = ItemCollaborativeModel(movies, ratings)
        # self.collaborative_user = UserCollaborativeModel(movies, ratings)
        # self.hybrid = HybridModel(
        #     movies,
        #     self.content,
        #     self.collaborative_item
        # )
