# # src/models/collaborative.py
# import pandas as pd
# import numpy as np
# from typing import List, Tuple, Dict

# from src.models.base import BaseRecommender
# from src.features.similarity import build_user_item_matrix, compute_topk_similarity
# from utils.load_data import get_movie_id, get_movie_ids
# from src.types.rating import Rating


# class ItemCollaborativeModel(BaseRecommender):
#     def __init__(
#         self,
#         movies: pd.DataFrame,
#         ratings: pd.DataFrame,
#         k_items: int = 50,
#         k_users: int = 50
#     ):
#         super().__init__(movies)
#         self.ratings = ratings
#         self.k_items = k_items
#         self.k_users = k_users
#         self._user_item_matrix = None
#         self._item_neighbors = None

#     # ---------- Core structures ----------
#     @property
#     def user_item_matrix(self):
#         if self._user_item_matrix is None:
#             self._user_item_matrix = build_user_item_matrix(self.ratings)
#         return self._user_item_matrix

#     def _invalidate(self):
#         self._user_item_matrix = None
#         self._item_neighbors = None

#     # ---------- Item–item ----------
#     @property
#     def item_neighbors(self):
#         if self._item_neighbors is None:
#             uim = self.user_item_matrix
#             self._item_neighbors = compute_topk_similarity(
#                 uim.T.values,
#                 index_labels=uim.columns.tolist(),
#                 k=self.k_items
#             )
#         return self._item_neighbors

#     def recommend(self, title: str, top_n: int) -> pd.DataFrame:
#         movie_id = get_movie_id(title, self.movies)
#         neighbors = self.item_neighbors[movie_id][:top_n]
#         return self._movies_from_ids([mid for mid, _ in neighbors])
