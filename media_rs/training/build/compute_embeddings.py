import numpy as np
from media_rs.features.embeddings import compute_item_embeddings, compute_user_embeddings

def compute_item_and_user_embeddings(movies, user_item_matrix):
    # Item embeddings
    item_embeddings, vectorizer, svd = compute_item_embeddings(movies["content"].values)

    # User embeddings (sparse-aware)
    user_embeddings_dict = compute_user_embeddings(
        {uid: user_item_matrix[uid].indices.tolist() for uid in range(user_item_matrix.shape[0])},
        item_embeddings
    )
    user_embeddings = np.vstack([user_embeddings_dict[uid] for uid in range(user_item_matrix.shape[0])]).astype(np.float32)

    return item_embeddings, user_embeddings, vectorizer, svd
