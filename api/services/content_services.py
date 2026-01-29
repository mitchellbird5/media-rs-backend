from media_rs.serving.recommender.build.build_content_model import get_content_similarity_model
from media_rs.utils.item_index import ItemIndex
from media_rs.utils.movies.movie_data_cache import get_movie_data_cache
from media_rs.rs_types.model import EmbeddingMethod

from typing import List

def get_content_recommendations(
    movie_title: str,
    method: EmbeddingMethod,
    top_n: int = 10,
) -> List[str]:
    """
    Service function to get content-based recommendations.
    """
    cache = get_movie_data_cache()
    item_idx = ItemIndex(cache.get("item_index.pkl"))
    
    rs_content = get_content_similarity_model(cache, method)
    
    recommendations = rs_content.recommend(item_idx.title_to_idx(movie_title), top_n)
    return [item_idx.idx_to_title(r[0]) for r in recommendations]

def get_content_recommendations_from_description(
    description: str,
    method: EmbeddingMethod,
    top_n: int = 10
) -> List[str]:
    """
    Service function to get content-based recommendations.
    """
    cache = get_movie_data_cache()
    item_idx = ItemIndex(cache.get("item_index.pkl"))
    
    rs_content = get_content_similarity_model(cache, method)
    
    recommendations = rs_content.recommend_from_description(description, top_n)
    return [item_idx.idx_to_title(r[0]) for r in recommendations]