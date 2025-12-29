import numpy as np
import faiss

from scipy.sparse import csr_matrix

from typing import List, Dict, Tuple, Union
from media_rs.rs_types.model import IdType, ContentSimilarity

class ItemItemCollaborativeModel:
    def __init__(
        self, 
        ids: List[IdType],
        topk_graph:  Dict[int, List[Tuple[int, float]]],
    ):
        self.ids = ids
        self.id_to_idx = {i: idx for idx, i in enumerate(ids)}
        self.topk_graph = topk_graph

    def recommend(self, item_id, top_n):
        idx = self.id_to_idx[item_id]
        return self.topk_graph[idx][:top_n]
    
class UserCollaborativeModel:
    def __init__(
        self,
        faiss_index: faiss.Index,
        user_item_matrix: csr_matrix        # shape (num_users, num_items)
    ):
        self.index = faiss_index
        self.user_item_matrix = user_item_matrix
        self.num_items = user_item_matrix.shape[1]

    def recommend(
        self,
        ratings: Dict[int, float],
        item_embeddings: np.ndarray,
        top_n: int = 10,
        k_similar_users: int = 50
    ) -> List[ContentSimilarity]:
        """
        Recommend items for a new user given a dict of {item_idx: rating}
        """
        # Compute new user embedding in same space as existing users
        user_emb = np.zeros(item_embeddings.shape[1], dtype=np.float32)
        for item_idx, rating in ratings.items():
            user_emb += rating * item_embeddings[item_idx]
        user_emb = user_emb[None, :]  # 2D
        faiss.normalize_L2(user_emb)

        # Mask items already rated
        rated_mask = np.zeros(self.num_items, dtype=bool)
        rated_mask[list(ratings.keys())] = True

        return self._recommend_from_embedding(user_emb, rated_mask, top_n, k_similar_users)


    def _recommend_from_embedding(
        self,
        user_emb: np.ndarray,
        rated_mask: Union[np.ndarray, csr_matrix],
        top_n: int,
        k_similar_users: int
    ) -> List[ContentSimilarity]:

        # Normalize for cosine similarity
        faiss.normalize_L2(user_emb)

        # Query FAISS
        D, I = self.index.search(user_emb, k_similar_users)
        similar_users = I[0]
        similarities = D[0]

        # ---- SPARSE AGGREGATION ----
        # Accumulate weighted rows into a sparse vector
        agg_scores = None  # csr_matrix (1, num_items)

        for sim_user_idx, sim in zip(similar_users, similarities):
            if sim <= 0:
                continue

            row = self.user_item_matrix.getrow(sim_user_idx)
            if row.nnz == 0:
                continue

            weighted_row = row.multiply(sim)

            if agg_scores is None:
                agg_scores = weighted_row
            else:
                agg_scores += weighted_row

        if agg_scores is None:
            return []

        # Convert ONCE to dense
        scores = agg_scores.toarray().ravel()

        # ---- MASK ALREADY RATED ITEMS ----
        if isinstance(rated_mask, csr_matrix):
            scores[rated_mask.indices] = -np.inf
        else:
            scores[rated_mask] = -np.inf

        # ---- TOP-N ----
        top_indices = np.argpartition(-scores, top_n)[:top_n]
        top_indices = top_indices[np.argsort(-scores[top_indices])]

        return [(int(i), float(scores[i])) for i in top_indices]