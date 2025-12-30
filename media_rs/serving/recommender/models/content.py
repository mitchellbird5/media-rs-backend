# src/models/content.py
import numpy as np

from typing import List, Dict, Tuple, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

from media_rs.rs_types.model import IdType, ContentSimilarity


class ContentModel:
    """
    Recommendation system model based on content similarity.
    """
    def __init__(
        self,
        topk_graph:  Dict[int, List[Tuple[int, float]]],
        embeddings: np.ndarray,
        vectorizer: TfidfVectorizer,
        svd: TruncatedSVD
    ):
        """
        Initialisation

        Args:
            topk_graph (Dict[int, List[ContentSimilarity]]): 
                Top K most similar neighbours of each item
            
            embeddings (np.ndarray): 
                Item embedding matrix of shape (num_items, embedding_dim),
                where each row represents an item in a latent vector space.
                
            vectorizer (TfidfVectorizer): 
                Fitted TF-IDF vectorizer used to convert raw text into a 
                sparse term-frequency representation.
            
            svd (TruncatedSVD): 
                Fitted dimensionality-reduction model that projects TF-IDF vectors into a
                dense latent embedding space.
        """
        
        self.topk_graph = topk_graph
        self.embeddings = embeddings
        self.vectorizer = vectorizer
        self.svd = svd

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

    def recommend_from_text(
        self, 
        text: str, 
        top_n: int
    ) -> List[ContentSimilarity]:
        """
        Recommend n most similar items to text description

        Args:
            text (str): Description used to compare
            top_n (int): Number of results to return

        Returns:
            List[ContentSimilarity]: 
                List of results.
                Tuple of ID of item and similarity score for each result in list
        """

        vec = self.vectorizer.transform([text])
        emb = self.svd.transform(vec).astype("float32")

        sims = emb @ self.embeddings.T

        sims = sims.ravel()

        # Top-N indices
        top_indices = np.argsort(-sims)[:top_n]

        # Return (index, score)
        return [(int(i), float(sims[i])) for i in top_indices]