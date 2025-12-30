import numpy as np

from typing import List, Dict, Tuple, Optional

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sentence_transformers import SentenceTransformer

from media_rs.rs_types.model import IdType, ContentSimilarity


class ContentSimilarityModel:
    """
    Recommendation system model based on content similarity.
    """
    def __init__(
        self,
        topk_graph:  Dict[int, List[Tuple[int, float]]],
        embeddings: np.ndarray,
        transformer: SentenceTransformer,
    ):
        """
        Initialisation

        Args:
            topk_graph (Dict[int, List[ContentSimilarity]]): 
                Top K most similar neighbours of each item
            
            embeddings (np.ndarray): 
                Item embedding matrix of shape (num_items, embedding_dim),
                where each row represents an item in a latent vector space.
                
            transformer (SentenceTransformer): 
                Fitted SBERT sentence transformer.
        """
        
        self.topk_graph = topk_graph
        self.embeddings = embeddings
        self.transformer = transformer

    def recommend(
        self, 
        item_id: IdType, 
        top_n: int
    ) -> List[ContentSimilarity]:
        """
        Recommend n most similar items for item_id

        Args:
            item_id (IdType): ID of item to compare
            top_n (int): Number of results to return

        Returns:
            List[ContentSimilarity]: 
                List of results.
                Tuple of ID of item and similarity score for each result in list
        """
        
        return self.topk_graph[item_id][:top_n]
    
    def recommend_from_description(
        self, 
        description: str, 
        top_n: int
    ) -> List[ContentSimilarity]:
        """
        Recommend n most similar items for item_id

        Args:
            description (IdType): Description to compare against
            top_n (int): Number of results to return

        Returns:
            List[ContentSimilarity]: 
                List of results.
                Tuple of ID of item and similarity score for each result in list
        """
        # Compute SBERT embedding
        emb = self.transformer.encode([description], convert_to_numpy=True, normalize_embeddings=True)
        
        # Compute cosine similarity
        sims = emb @ self.embeddings.T  # (1, num_items)
        sims = sims.ravel()
        
        # Top-N indices
        top_indices = np.argsort(-sims)[:top_n]
        
        return [(int(i), float(sims[i])) for i in top_indices]