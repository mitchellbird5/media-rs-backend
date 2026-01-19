import numpy as np
from typing import List
from scipy.sparse import csr_matrix

from media_rs.training.features.embeddings import (
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
    user_embeddings = compute_user_embeddings(
        user_item_matrix,
        item_embeddings,
    )
    return item_embeddings, sbert_model, user_embeddings
