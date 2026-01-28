import faiss
import numpy as np

from typing import Tuple

from media_rs.training.features.faiss import build_faiss_index, FaissMethod

def build_faiss_indices(
    item_embeddings: np.ndarray, 
    user_embeddings: np.ndarray
) -> Tuple[faiss.Index, faiss.Index]:
    """
    Returns FAISS indices of item and user embeddings

    Args:
        item_embeddings (np.ndarray): Item embeddings
        user_embeddings (np.ndarray): User embeddings

    Returns:
        _type_: _description_
    """
    
    faiss_index_content = build_faiss_index(item_embeddings, metric=FaissMethod.COSINE)

    faiss_index_users = faiss.IndexFlatIP(user_embeddings.shape[1])
    faiss_index_users.add(user_embeddings)

    return faiss_index_content, faiss_index_users
