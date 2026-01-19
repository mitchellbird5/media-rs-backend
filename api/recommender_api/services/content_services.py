from media_rs.serving.recommender.build.build_content_model import get_content_similarity_model
from media_rs.utils.item_index import ItemIndex
from media_rs.rs_types.model import ContentSimilarity
from media_rs.utils.movies.movie_data_cache import get_movie_data_cache

from typing import List

cache = get_movie_data_cache()

def get_content_recommendations(
    movie_title: str,
    top_n: int = 10
) -> List[str]:
    """
    Service function to get content-based recommendations.
    """
    
    item_idx = ItemIndex(cache.get("item_index.pkl"))
    
    rs_content = get_content_similarity_model(cache)
    
    idx = item_idx.title_to_idx(movie_title)
    if len(idx) > 1:
        raise ValueError(f"Multiple titles found that match '{movie_title}'")
    elif len(idx) != 1:
        raise ValueError(f"No titles found that match '{movie_title}'")
    else: 
        index = idx[0]
    
    recommendations = rs_content.recommend(index, top_n)
    return [item_idx.idx_to_title(r[0]) for r in recommendations]

def get_content_recommendations_from_description(
    description: str,
    top_n: int = 10
) -> List[str]:
    """
    Service function to get content-based recommendations.
    """
    item_idx = ItemIndex(cache.get("item_index.pkl"))
    
    rs = get_content_similarity_model(cache)
    
    recommendations = rs.recommend_from_description(description, top_n)
    return [item_idx.idx_to_title(r[0]) for r in recommendations]

