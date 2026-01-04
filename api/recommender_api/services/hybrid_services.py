import numpy as np

from pathlib import Path
from typing import Dict, List

from media_rs.utils.item_index import ItemIndex
from media_rs.serving.recommender.build.build_hybrid_model import get_hybrid_model
from media_rs.rs_types.model import ContentSimilarity

from media_rs.utils.movies.movie_data_cache import get_movie_data_cache

cache = get_movie_data_cache()

def get_hybrid_recommendations(
    movie_title: str,
    ratings: Dict[str, float],
    alpha: float,
    beta: float,
    top_n: int,
    k_similar_users: int
) -> List[ContentSimilarity]:

    item_idx = ItemIndex(cache.get("item_index.pkl"))
    item_embeddings = cache.load("movies_item_embeddings.npy")
    
    index_ratings = {
        item_idx.title_to_idx[title]: score
        for title, score in ratings.items()
        if title in item_idx.title_to_idx
    }
    
    rs = get_hybrid_model(item_idx, alpha, beta)
    
    recommendations = rs.recommend(
        item_idx.title_to_idx[movie_title], 
        index_ratings,
        item_embeddings,
        k_similar_users,
        top_n,
    )
    return [item_idx.idx_to_title[r[0]] for r in recommendations]