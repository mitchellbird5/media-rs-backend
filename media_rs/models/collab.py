import numpy as np
import faiss

from typing import List, Dict, Tuple, Optional
from media_rs.models.base import BaseRecommender
from media_rs.rs_types.model import IdType, ContentSimilarity

class ItemItemCollaborativeModel(BaseRecommender):
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
    
class UserCollaborativeModel(BaseRecommender):
    def __init__(
        self, 
        user_embeddings: np.ndarray,      # shape (num_users, embedding_dim)
        faiss_index: faiss.Index,
        user_item_matrix: np.ndarray,     # shape (num_users, num_items)
    ):
        self.user_embeddings = user_embeddings.astype(np.float32)
        self.index = faiss_index
        self.user_item_matrix = user_item_matrix

    def recommend(self, user_idx: int, top_n: int = 10, k_similar_users: int = 50) -> List[ContentSimilarity]:
        # Select user embedding as 2D array
        user_emb = self.user_embeddings[user_idx:user_idx+1]

        # Normalize for cosine similarity
        faiss.normalize_L2(user_emb)

        # Query FAISS index for similar users
        D, I = self.index.search(user_emb, k_similar_users)

        similar_users = I[0]
        similarities = D[0]

        # Aggregate ratings from similar users
        scores = np.zeros(self.user_item_matrix.shape[1], dtype=np.float32)
        for sim_user_idx, sim in zip(similar_users, similarities):
            scores += sim * self.user_item_matrix[sim_user_idx]

        # Mask items the target user already rated
        rated = self.user_item_matrix[user_idx] > 0
        scores[rated] = -np.inf

        # Get top-N item indices
        top_indices = np.argsort(-scores)[:top_n]

        # Convert to (movieId, score) tuples
        return [(i, float(scores[i])) for i in top_indices]