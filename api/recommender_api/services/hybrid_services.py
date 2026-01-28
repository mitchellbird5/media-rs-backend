from typing import Dict, List

from media_rs.utils.item_index import ItemIndex
from media_rs.serving.recommender.build.build_hybrid_model import (
    get_hybrid_model,
    get_hybrid_embeddings
)
from media_rs.utils.movies.movie_data_cache import get_movie_data_cache
from media_rs.rs_types.rating import get_index_ratings
from media_rs.rs_types.model import EmbeddingMethod

def get_hybrid_recommendations(
    movie_title: str,
    ratings: Dict[str, float],
    alpha: float,
    beta: float,
    top_n: int,
    k_similar_users: int,
    method: EmbeddingMethod
) -> List[str]:
    cache = get_movie_data_cache()
    item_idx = ItemIndex(cache.get("item_index.pkl"))
    
    index_ratings = get_index_ratings(ratings, item_idx)
    if not index_ratings:
        raise ValueError("No valid user ratings after normalization")
    
    rs = get_hybrid_model(
        cache,
        method,
        alpha, 
        beta
    )
    
    recommendations = rs.recommend(
        item_idx.title_to_idx(movie_title), 
        index_ratings,
        k_similar_users,
        top_n,
    )
    return [item_idx.idx_to_title(r[0]) for r in recommendations]