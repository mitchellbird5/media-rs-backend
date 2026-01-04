from media_rs.serving.recommender.models.collab import (
    ItemItemCollaborativeModel,
    UserCollaborativeModel,
)
from media_rs.utils.movies.movie_data_cache import MovieDataCache

def get_item_cf_model(cache: MovieDataCache) -> ItemItemCollaborativeModel:
    return ItemItemCollaborativeModel(
        topk_graph=cache.get("movies_item_topk_cf.pkl")
    )
    
def get_user_cf_model(cache: MovieDataCache) -> UserCollaborativeModel:
    return UserCollaborativeModel(
        faiss_index=cache.get("faiss_index_users.index"),
        user_item_matrix=cache.get("user_item_matrix.npz"),
        item_embeddings=cache.get("movies_item_embeddings.npy")
    )
    