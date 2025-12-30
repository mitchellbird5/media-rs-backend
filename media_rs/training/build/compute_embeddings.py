import numpy as np
from typing import List
from scipy.sparse import csr_matrix

from media_rs.training.features.embeddings import (
    compute_item_embeddings, 
    compute_user_embeddings,
    compute_sbert_embeddings
)

def compute_item_and_user_embeddings(
    content: List[str], 
    user_item_matrix: csr_matrix
):
    # Item embeddings
    # item_embeddings, vectorizer, svd = compute_item_embeddings(content)
    
    sbert_model, item_embeddings = compute_sbert_embeddings(content)

    # User embeddings (sparse-aware)
    user_embeddings_dict = compute_user_embeddings(
        {uid: user_item_matrix[uid].indices.tolist() for uid in range(user_item_matrix.shape[0])},
        item_embeddings
    )
    user_embeddings = np.vstack([user_embeddings_dict[uid] for uid in range(user_item_matrix.shape[0])]).astype(np.float32)

    return item_embeddings, sbert_model, user_embeddings
