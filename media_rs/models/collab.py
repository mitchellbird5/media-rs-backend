import numpy as np
import faiss

from typing import List, Dict, Tuple, Optional
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
        user_embeddings: np.ndarray,        # shape (num_users, embedding_dim)
        faiss_index: faiss.Index,
        user_item_matrix: np.ndarray        # shape (num_users, num_items)
    ):
        self.user_embeddings = user_embeddings.astype(np.float32)
        self.index = faiss_index
        self.user_item_matrix = user_item_matrix
        self.num_items = user_item_matrix.shape[1]
        self.embedding_dim = user_embeddings.shape[1]

    def recommend_existing_user(
        self,
        user_idx: int,
        top_n: int = 10,
        k_similar_users: int = 50
    ) -> List[ContentSimilarity]:
        """
        Recommend items for an existing user by index
        """
        user_emb = self.user_embeddings[user_idx:user_idx+1]  # keep 2D
        rated_mask = self.user_item_matrix[user_idx]
        return self._recommend_from_embedding(user_emb, rated_mask, top_n, k_similar_users)

    def recommend_from_ratings(
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
        rated_mask: np.ndarray,
        top_n: int,
        k_similar_users: int
    ) -> List[ContentSimilarity]:
        """
        Core FAISS + aggregation logic
        """
        # Normalize input embedding (cosine similarity)
        faiss.normalize_L2(user_emb)

        # Query FAISS for similar users
        D, I = self.index.search(user_emb, k_similar_users)
        similar_users = I[0]
        similarities = D[0]

        # Aggregate scores from similar users
        scores = np.zeros(self.num_items, dtype=np.float32)
        for sim_user_idx, sim in zip(similar_users, similarities):
            scores += sim * self.user_item_matrix[sim_user_idx]

        # Mask already rated items
        scores[rated_mask > 0] = -np.inf

        # Return top-N items as (item_idx, score)
        top_indices = np.argsort(-scores)[:top_n]
        return [(i, float(scores[i])) for i in top_indices]