from media_rs.utils.item_index import ItemIndex
from media_rs.rs_types.model import ContentSimilarity
from media_rs.serving.recommender.build.build_collab_model import (
    get_item_cf_model,
    get_user_cf_model
)
from media_rs.utils.movies.movie_data_cache import get_movie_data_cache

from typing import List, Dict

cache = get_movie_data_cache()

def get_item_cf_recommendations(
    movie_title: str,
    top_n: int = 10
) -> List[ContentSimilarity]:
    
    item_idx = ItemIndex(cache.get("item_index.pkl"))
    
    rs = get_item_cf_model(cache)
    
    recommendations = rs.recommend(item_idx.title_to_idx[movie_title], top_n)
    return [item_idx.idx_to_title[r[0]] for r in recommendations]

def get_user_cf_recommendations(
    ratings: Dict[str, float],
    top_n: int,
    k_similar_users: int
) -> List[ContentSimilarity]:
    
    item_idx = ItemIndex(cache.get("item_index.pkl"))
    
    index_ratings = {
        item_idx.title_to_idx[title]: score
        for title, score in ratings.items()
        if title in item_idx.title_to_idx
    }
    
    rs = get_user_cf_model(cache)
    
    recommendations = rs.recommend(
        index_ratings, 
        top_n,
        k_similar_users
    )
    return [item_idx.idx_to_title[r[0]] for r in recommendations]