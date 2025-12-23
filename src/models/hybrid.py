# # src/models/hybrid.py
# import pandas as pd

# from src.models.base import BaseRecommender
# from src.models.content import ContentModel
# from src.models.item_collab import ItemCollaborativeModel
# from utils.load_data import get_movie_id


# class HybridModel(BaseRecommender):
#     def __init__(
#         self,
#         movies: pd.DataFrame,
#         content_model: ContentModel,
#         collaborative_model: ItemCollaborativeModel,
#         alpha: float = 0.7
#     ):
#         super().__init__(movies)
#         self.content_model = content_model
#         self.collaborative_model = collaborative_model
#         self.alpha = alpha

#     def recommend(self, title: str, top_n: int) -> pd.DataFrame:
#         movie_id = get_movie_id(title, self.movies)

#         content = dict(self.content_model.neighbors[movie_id])
#         item = dict(self.collaborative_model.item_neighbors[movie_id])

#         all_ids = set(content) | set(item)

#         scores = {
#             mid: self.alpha * content.get(mid, 0.0)
#             + (1 - self.alpha) * item.get(mid, 0.0)
#             for mid in all_ids
#         }

#         top_ids = (
#             pd.Series(scores)
#             .sort_values(ascending=False)
#             .head(top_n)
#             .index.tolist()
#         )

#         return self._movies_from_ids(top_ids)
