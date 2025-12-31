from media_rs.serving.recommender.models.collab import (
    ItemItemCollaborativeModel,
    UserCollaborativeModel,
)
from media_rs.utils.movies.movie_data_cache import MovieDataCache

def get_item_cf_model(cache: MovieDataCache) -> ItemItemCollaborativeModel:
    return ItemItemCollaborativeModel(
        topk_graph=cache.load("movies_item_topk_cf.pkl")
    )
    
def get_user_cf_model(cache: MovieDataCache) -> UserCollaborativeModel:
    return UserCollaborativeModel(
        faiss_index=cache.load("faiss_index_users.index"),
        user_item_matrix=cache.load("user_item_matrix.npz"),
        item_embeddings=cache.load("movies_item_embeddings.npy")
    )
    