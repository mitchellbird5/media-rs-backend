from media_rs.serving.recommender.models.collab import (
    ItemItemCollaborativeModel,
    UserCollaborativeModel,
)
from media_rs.utils.data_cache import DataCache
from media_rs.rs_types.model import EmbeddingMethod, Medium

def get_item_cf_model(
    cache: DataCache,
    medium: Medium
) -> ItemItemCollaborativeModel:
    return ItemItemCollaborativeModel(
        topk_graph=cache.get(f"{medium.value}/item_topk_cf.pkl")
    )
    
def get_user_sbert_cf_model(
    cache: DataCache,
    medium: Medium
) -> UserCollaborativeModel:
    return UserCollaborativeModel(
        faiss_index=cache.get(f"{medium.value}/sbert/faiss_index_users.index"),
        user_item_matrix=cache.get(f"{medium.value}/user_item_matrix.npz"),
        item_embeddings=cache.get(f"{medium.value}/sbert/item_embeddings.npy")
    )

def get_user_tfidf_cf_model(
    cache: DataCache,
    medium: Medium  
) -> UserCollaborativeModel:
    return UserCollaborativeModel(
        faiss_index=cache.get(f"{medium.value}/tfidf/faiss_index_users.index"),
        user_item_matrix=cache.get(f"{medium.value}/user_item_matrix.npz"),
        item_embeddings=cache.get(f"{medium.value}/tfidf/item_embeddings.npy")
    )
    
def get_user_cf_model(
    cache: DataCache,
    method: EmbeddingMethod,
    medium: Medium
) -> UserCollaborativeModel:
    if method==EmbeddingMethod.SBERT:
        return get_user_sbert_cf_model(cache, medium)
    if method==EmbeddingMethod.TFIDF:
        return get_user_tfidf_cf_model(cache, medium)