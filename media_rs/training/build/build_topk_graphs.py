import numpy as np
import faiss
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix

from typing import List

from media_rs.rs_types.model import ContentSimilarity

def build_item_cf_topk(
    user_item_matrix: csr_matrix, 
    k: int=100, 
    batch_size: int=1000
) -> List[ContentSimilarity]:
    """
    Builds similarity graph of nearest K users using cosine similarity.

    Args:
        user_item_matrix (csr_matrix): 
            Sparse user–item interaction matrix of shape
            (num_users, num_items), where rows correspond to users
            and columns correspond to items.
        
        k (int, optional): 
            Number of nearest users to graph. Defaults to 100.
        
        batch_size (int, optional): 
            Size of batch to compute at any one time. Defaults to 1000.

    Returns:
        List[ContentSimilarity]: Top K nearest neighbour graph
    """
    
    num_items = user_item_matrix.shape[1]
    topk_cf = {}

    for start in range(0, num_items, batch_size):
        end = min(start + batch_size, num_items)
        batch_matrix = user_item_matrix[:, start:end].T
        sim = cosine_similarity(batch_matrix, user_item_matrix.T)
        for i, item_idx in enumerate(range(start, end)):
            top_indices = np.argsort(-sim[i])[:k+1]
            topk_cf[item_idx] = [(j, sim[i,j]) for j in top_indices if j != item_idx]
    return topk_cf


def build_topk_content(
    item_embeddings:np.ndarray, 
    top_k: int=100
) -> List[ContentSimilarity]:
    """
    Builds similarity graph of nearest K users using FAISS.

    Args:
        item_embeddings (np.ndarray): 
            Item embedding matrix of shape (num_items, embedding_dim),
            where each row represents an item in a latent vector space.
        
        top_k (int, optional): Number of nearest users to graph. Defaults to 100.

    Returns:
        List[ContentSimilarity]: Top K nearest neighbour graph
    """
    
    # Normalize embeddings for cosine similarity
    faiss.normalize_L2(item_embeddings)
    
    # Create FAISS index
    dim = item_embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # Inner product = cosine if vectors normalized
    index.add(item_embeddings)
    
    # Search top-k for each item
    distances, indices = index.search(item_embeddings, top_k + 1)  # +1 because the first is itself
    
    # Build mapping: item_idx -> top_k item_idx
    topk_content = {}
    for i, neighbors in enumerate(indices):
        topk_content[i] = [(n, d) for n, d in zip(neighbors[1:], distances[i][1:])]  # skip self
    
    return topk_content