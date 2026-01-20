from media_rs.serving.recommender.models.collab import (
    ItemItemCollaborativeModel,
    UserCollaborativeModel,
)
from media_rs.utils.movies.movie_data_cache import MovieDataCache
from media_rs.rs_types.model import EmbeddingMethod

def get_item_cf_model(cache: MovieDataCache) -> ItemItemCollaborativeModel:
    return ItemItemCollaborativeModel(
        topk_graph=cache.get("item_topk_cf.pkl")
    )
    
def get_user_sbert_cf_model(cache: MovieDataCache) -> UserCollaborativeModel:
    return UserCollaborativeModel(
        faiss_index=cache.get("sbert/faiss_index_users.index"),
        user_item_matrix=cache.get("user_item_matrix.npz"),
        item_embeddings=cache.get("sbert/item_embeddings.npy")
    )

def get_user_tfidf_cf_model(cache: MovieDataCache) -> UserCollaborativeModel:
    return UserCollaborativeModel(
        faiss_index=cache.get("tfidf/faiss_index_users.index"),
        user_item_matrix=cache.get("user_item_matrix.npz"),
        item_embeddings=cache.get("tfidf/item_embeddings.npy")
    )
    
def get_user_cf_model(
    cache: MovieDataCache,
    method: EmbeddingMethod
) -> UserCollaborativeModel:
    if method==EmbeddingMethod.SBERT:
        return get_user_sbert_cf_model(cache)
    if method==EmbeddingMethod.TFIDF:
        return get_user_tfidf_cf_model(cache)