import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Tuple

def compute_item_embeddings(
    item_texts: List[str],
    n_features: int = 50000,
    n_components: int = 200,
    vectorizer=None,
    svd=None
) -> Tuple[np.ndarray, TfidfVectorizer, TruncatedSVD]:

    """
    Embeds list of text using TF-IDF and SVD
    
    Args:
        item_texts (List[str]): 
            List of texts to embed
            
        n_features (int): 
            Max feature limit to use in TFIDF. Defaults to 50000.
            
        n_components (int): 
            Number of components to use in SVD

    Returns:
        Tuple[np.ndarray, TfidfVectorizer, TruncatedSVD]: 
            1. Item embeddings
            2. TFIDF Vectoriser
            3. SVD
    """

    if vectorizer is None:
        vectorizer = TfidfVectorizer(max_features=n_features)
        tfidf_matrix = vectorizer.fit_transform(item_texts)
    else:
        tfidf_matrix = vectorizer.transform(item_texts)
        
    if svd is None:
        svd = TruncatedSVD(n_components=n_components, random_state=42)
        embeddings = svd.fit_transform(tfidf_matrix)
    else:
        embeddings = svd.transform(tfidf_matrix)
    
    return np.asarray(embeddings, dtype=np.float32), vectorizer, svd


def compute_user_embeddings(
    user_item_dict: Dict[int, List[int]],
    item_embeddings: np.ndarray
) -> Dict[int, np.ndarray]:
    """
    Compute embeddings for user ratings

    Args:
        user_item_dict (Dict[int, List[int]]): Dictionary of userID to other user indices
        item_embeddings (np.ndarray): Embeddings of items

    Returns:
        Dict[int, np.ndarray]: User embeddings
    """

    user_embeddings = {}
    for user_id, item_ids in user_item_dict.items():
        vectors = item_embeddings[item_ids]
        user_embeddings[user_id] = np.mean(vectors, axis=0)
    return user_embeddings
