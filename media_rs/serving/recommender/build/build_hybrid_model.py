import numpy as np

from media_rs.serving.recommender.models.hybrid import HybridModel
from media_rs.utils.item_index import ItemIndex
from media_rs.serving.recommender.build.build_collab_model import (
    get_item_cf_model, 
    get_user_cf_model
)
from media_rs.serving.recommender.build.build_content_model import get_content_similarity_model
from media_rs.rs_types.model import EmbeddingMethod
from media_rs.utils.movies.movie_data_cache import MovieDataCache


def get_hybrid_model(
    cache: MovieDataCache,
    method: EmbeddingMethod,
    alpha: float,
    beta: float
) -> HybridModel:
    
    """Retrieves hybrid filter model.

    Args:
        item_index (ItemIndex): ItemIndex object of index data.

    Returns:
        HybridModel: Hybrid filter model
    """
    
    content_model = get_content_similarity_model(cache, method)
    item_cf_model = get_item_cf_model(cache)
    user_cf_model = get_user_cf_model(cache, method)
    
    return HybridModel(
        content_model=content_model,
        item_collab_model=item_cf_model,
        user_collab_model=user_cf_model,
        alpha=alpha,
        beta=beta
    )
    
def get_hybrid_embeddings(
    cache: MovieDataCache,
    method: EmbeddingMethod
) -> np.ndarray:
    if method == EmbeddingMethod.SBERT:
        return cache.get("sbert/item_embeddings.npy")
    if method == EmbeddingMethod.TFIDF:
        return cache.get("tfidf/item_embeddings.npy")