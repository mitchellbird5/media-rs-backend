from media_rs.utils.item_index import ItemIndex
from media_rs.rs_types.model import ContentSimilarity
from media_rs.serving.recommender.build.build_collab_model import (
    get_item_cf_model,
    get_user_cf_model
)
from media_rs.utils.movies.movie_data_cache import get_movie_data_cache
from media_rs.utils.movies.build_user_item_matrix import norm
from media_rs.rs_types.rating import Rating, get_index_ratings

from typing import List, Dict

cache = get_movie_data_cache(local_dir='data/movies/raw/ml-latest-small/cache')

def get_item_cf_recommendations(
    movie_title: str,
    top_n: int = 10
) -> List[ContentSimilarity]:
    
    item_idx = ItemIndex(cache.get("item_index.pkl"))
    
    rs = get_item_cf_model(cache)
    
    idx = item_idx.title_to_idx(movie_title)
    if len(idx) > 1:
        raise ValueError(f"Multiple titles found that match '{movie_title}'")
    elif len(idx) != 1:
        raise ValueError(f"No titles found that match '{movie_title}'")
    else: 
        index = idx[0]
    
    recommendations = rs.recommend(index, top_n)
    return [item_idx.idx_to_title(r[0]) for r in recommendations]

def get_user_cf_recommendations(
    ratings: List[Rating],
    top_n: int,
    k_similar_users: int
) -> List[str]:
    
    item_idx = ItemIndex(cache.get("item_index.pkl"))

    index_ratings = get_index_ratings(ratings, item_idx)
    if not index_ratings:
        raise ValueError("No valid user ratings after normalization")

    rs = get_user_cf_model(cache)

    recommendations = rs.recommend(
        index_ratings,
        top_n,
        k_similar_users
    )
    
    return [item_idx.idx_to_title(r[0]) for r in recommendations]