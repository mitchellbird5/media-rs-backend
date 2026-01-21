import numpy as np
import faiss

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sentence_transformers import SentenceTransformer
from scipy.sparse import csr_matrix

from typing import List, Dict, Tuple

def compute_sbert_embeddings(
    item_texts: List[str]
) -> Tuple[SentenceTransformer, np.ndarray]:
    """
    Compute text embeddings using SBERT.

    Args:
        item_texts (List[str]): List of content to embed

    Returns:
        Tuple[SentenceTransformer, np.ndarray]: Tuple of model and embedding
    """
    
    sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
    return sbert_model, sbert_model.encode(item_texts, convert_to_numpy=True)
    

def compute_tfidf_embeddings(
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
    
    embeddings = np.asarray(embeddings, dtype=np.float32)
    faiss.normalize_L2(embeddings)
    
    return embeddings, vectorizer, svd


def compute_user_embeddings(
    user_item_matrix: csr_matrix,
    item_embeddings: np.ndarray,
) -> np.ndarray:

    num_users = user_item_matrix.shape[0]
    emb_dim = item_embeddings.shape[1]

    user_embeddings = np.zeros((num_users, emb_dim), dtype=np.float32)

    for uid in range(num_users):
        row = user_item_matrix.getrow(uid)

        if row.nnz == 0:
            continue

        item_ids = row.indices
        ratings = row.data.astype(np.float32)

        ratings = ratings - ratings.mean()

        vectors = item_embeddings[item_ids] * ratings[:, None]
        user_embeddings[uid] = vectors.mean(axis=0)

    faiss.normalize_L2(user_embeddings)
    return user_embeddings
