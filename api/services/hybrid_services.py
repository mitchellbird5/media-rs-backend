from typing import Dict, List

from media_rs.utils.item_index import ItemIndex
from media_rs.serving.recommender.build.build_hybrid_model import (
    get_hybrid_model,
    get_hybrid_embeddings
)
from media_rs.utils.data_cache import get_data_cache
from media_rs.rs_types.rating import get_index_ratings
from media_rs.rs_types.model import EmbeddingMethod, Medium

def get_hybrid_recommendations(
    title: str,
    ratings: Dict[str, float],
    alpha: float,
    beta: float,
    top_n: int,
    k_similar_users: int,
    method: EmbeddingMethod,
    medium: Medium
) -> List[str]:
    cache = get_data_cache()
    item_idx = ItemIndex(cache.get(f"{medium.value}/item_index.pkl"))
    
    index_ratings = get_index_ratings(ratings, item_idx)
    if not index_ratings:
        raise ValueError("No valid user ratings after normalization")
    
    rs = get_hybrid_model(
        cache=cache,
        method=method,
        medium=medium,
        alpha=alpha, 
        beta=beta
    )
    
    recommendations = rs.recommend(
        item_idx.title_to_idx(title), 
        index_ratings,
        k_similar_users,
        top_n,
    )
    return [item_idx.idx_to_title(r[0]) for r in recommendations]