# # src/models/collaborative.py
# import pandas as pd
# import numpy as np
# from typing import List, Tuple, Dict

# from src.models.base import BaseRecommender
# from src.features.similarity import build_user_item_matrix, compute_topk_similarity
# from utils.load_data import get_movie_id, get_movie_ids
# from src.types.rating import Rating


# class UserCollaborativeModel(BaseRecommender):
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

#     # ---------- Core structures ----------
#     @property
#     def user_item_matrix(self):
#         if self._user_item_matrix is None:
#             self._user_item_matrix = build_user_item_matrix(self.ratings)
#         return self._user_item_matrix

#     def _invalidate(self):
#         self._user_item_matrix = None

#     # ---------- User–user ----------
#     def recommend(
#         self,
#         ratings: List[Rating],
#         top_n: int
#     ) -> pd.DataFrame:
#         temp_user = self._build_temp_user_vector(ratings)
#         combined = pd.concat([self.user_item_matrix, temp_user])

#         neighbors = compute_topk_similarity(
#             combined.values,
#             index_labels=combined.index.tolist(),
#             k=self.k_users
#         )

#         sim_scores = dict(neighbors["temp"])
#         sim_series = pd.Series(sim_scores)

#         neighbor_ids = sim_series.index.tolist()

#         neighbor_matrix = self.user_item_matrix.loc[neighbor_ids]

#         weighted = (
#             neighbor_matrix.T.dot(sim_series)
#             / sim_series.sum()
#         )
#         already_rated = temp_user.iloc[0] > 0
#         weighted = weighted[~already_rated]

#         return self._movies_from_ids(
#             weighted.sort_values(ascending=False).head(top_n).index.tolist()
#         )

#     def _build_temp_user_vector(self, ratings: List[Rating]) -> pd.DataFrame:
#         movie_ids = get_movie_ids([r.title for r in ratings], self.movies)
#         row = pd.Series(0, index=self.user_item_matrix.columns, dtype=float)
#         for mid, r in zip(movie_ids, ratings):
#             row[mid] = r.rating
#         return row.to_frame().T.rename(index={0: "temp"})
