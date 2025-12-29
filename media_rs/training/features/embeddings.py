import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Tuple

# ----------------------------
# 1. Compute Item Embeddings
# ----------------------------
def compute_item_embeddings(
    item_texts: List[str],
    n_features: int = 50000,
    n_components: int = 200,
    vectorizer=None,
    svd=None
) -> Tuple[np.ndarray, TfidfVectorizer, TruncatedSVD]:
    """
    Convert a list of item texts to dense embeddings using TF-IDF + SVD
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

# ----------------------------
# 2. Compute User Embeddings
# ----------------------------
def compute_user_embeddings(
    user_item_dict: Dict[int, List[int]],
    item_embeddings: np.ndarray
) -> Dict[int, np.ndarray]:
    """
    Compute user embeddings as the mean of embeddings of items they rated
    """
    user_embeddings = {}
    for user_id, item_ids in user_item_dict.items():
        vectors = item_embeddings[item_ids]
        user_embeddings[user_id] = np.mean(vectors, axis=0)
    return user_embeddings

# ----------------------------
# 3. Compute Top-K Similarity Graph
# ----------------------------
def compute_topk_similarity_graph(
    embeddings: np.ndarray,
    k: int = 100
) -> Dict[int, List[Tuple[int, float]]]:
    """
    Compute top-K cosine similarity neighbors for each item
    """
    similarity_matrix = cosine_similarity(embeddings)
    topk_dict = {}
    for i, row in enumerate(similarity_matrix):
        top_k_idx = np.argpartition(-row, k)[:k]
        top_k_scores = row[top_k_idx]
        topk_dict[i] = list(zip(top_k_idx, top_k_scores))
    return topk_dict