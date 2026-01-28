import numpy as np
import faiss
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix

from typing import List

from media_rs.rs_types.model import ContentSimilarity

def build_item_cf_topk(
    user_item_matrix: csr_matrix,
    k: int = 100,
    batch_size: int = 1000
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

    if k <= 0:
        raise ValueError("k must be a positive integer")

    num_items = user_item_matrix.shape[1]

    if num_items < 2:
        raise ValueError("user_item_matrix must contain at least 2 items")

    # Clamp k to a safe maximum
    k = min(k, num_items - 1)

    topk_cf: dict[int, ContentSimilarity] = {}

    for start in range(0, num_items, batch_size):
        end = min(start + batch_size, num_items)

        # Shape: (batch_items, num_users)
        batch_matrix = user_item_matrix[:, start:end].T

        # Shape: (batch_items, num_items)
        sim = cosine_similarity(batch_matrix, user_item_matrix.T)

        for i, item_idx in enumerate(range(start, end)):
            # Get top k+1 to exclude self
            top_indices = np.argpartition(
                -sim[i], kth=k + 1
            )[: k + 1]

            neighbors = [
                (j, float(sim[i, j]))
                for j in top_indices
                if j != item_idx
            ]

            # Sort final neighbors
            neighbors.sort(key=lambda x: x[1], reverse=True)
            topk_cf[item_idx] = neighbors[:k]

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
    
    if top_k <= 0:
        raise ValueError("top_k must be a positive integer")

    if item_embeddings.ndim != 2:
        raise ValueError("item_embeddings must be a 2D array")

    num_items, dim = item_embeddings.shape

    if num_items < 2:
        raise ValueError("item_embeddings must contain at least 2 items")

    # Clamp top_k safely
    top_k = min(top_k, num_items - 1)

    # Normalize embeddings for cosine similarity
    item_embeddings = item_embeddings.astype(np.float32, copy=False)

    index = faiss.IndexFlatIP(dim)
    index.add(item_embeddings)

    # +1 to remove self-match
    distances, indices = index.search(item_embeddings, top_k + 1)

    topk_content: dict[int, ContentSimilarity] = {}

    for i in range(num_items):
        neighbors = [
            (int(n), float(d))
            for n, d in zip(indices[i], distances[i])
            if n != i
        ][:top_k]

        topk_content[i] = neighbors

    return topk_content