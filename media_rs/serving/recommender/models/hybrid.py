# src/models/hybrid.py
import pandas as pd
import numpy as np

from typing import List, Dict
from media_rs.rs_types.model import ContentSimilarity

from media_rs.serving.recommender.models.content import ContentModel
from media_rs.serving.recommender.models.collab import ItemItemCollaborativeModel, UserCollaborativeModel

class HybridModel:
    """
    Recommendation system model based on a hybrid of item-item and 
    user-user collaborative filtering, and content similarity.
    """
    def __init__(
        self,
        content_model: ContentModel,
        item_collab_model: ItemItemCollaborativeModel,
        user_collab_model: UserCollaborativeModel,
        alpha: float,
        beta: float
    ):
        """_summary_

        Args:
            content_model (ContentModel): 
                Recommendation system model based on content similarity.
            
            item_collab_model (ItemItemCollaborativeModel): 
                Recommendation system model based on item-item collaborative filtering.
            
            user_collab_model (UserCollaborativeModel): 
                Recommendation system model based on user-user collaborative filtering.
            
            alpha (float): 
                Weighting of content similarity score.
                
            beta (float): 
                Weighting of item collaborative filtering score.
        """
        
        self.content_model = content_model
        self.item_collab_model = item_collab_model
        self.user_collab_model = user_collab_model
        self.alpha = alpha
        self.beta = beta
        self.gamma = 1.0 - alpha - beta

    def recommend(
        self,
        item_idx: int,
        new_user_ratings: Dict[int, float],
        item_embeddings: np.ndarray,
        k_similar_users: int,
        top_n: int
    ) -> List[ContentSimilarity]:
        """
        Recommend n most similar items

        Args:
            item_idx (int): 
                ID of item to compare in content similarity model
            
            new_user_ratings (Dict[int, float]): 
                Ratings of movies to use in 
                user-user collaborative filtering.
            
            item_embeddings (np.ndarray): 
                Item embedding matrix of shape (num_items, embedding_dim),
                where each row represents an item in a latent vector space.
            
            k_similar_users (int, optional):
                Number of similar users to use in computation
            
            top_n (int, optional): 
                Number of results to return

        Returns:
            List[ContentSimilarity]: 
                List of results.
                Tuple of index of item and similarity score for each result
        """
        
        # 1. Content scores
        content_scores = dict(self.content_model.recommend(item_idx, top_n))

        # 2. Item-item CF scores
        item_scores = dict(self.item_collab_model.recommend(item_idx, top_n))

        # 3. User-user CF scores using new user ratings
        user_scores = dict(self.user_collab_model.recommend(
            new_user_ratings=new_user_ratings, 
            item_embeddings=item_embeddings, 
            top_n=top_n,
            k_similar_users=k_similar_users,
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