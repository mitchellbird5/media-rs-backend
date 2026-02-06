import numpy as np

from media_rs.serving.recommender.models.hybrid import HybridModel
from media_rs.serving.recommender.build.build_collab_model import (
    get_item_cf_model, 
    get_user_cf_model
)
from media_rs.serving.recommender.build.build_content_model import get_content_similarity_model
from media_rs.rs_types.model import EmbeddingMethod, Medium
from media_rs.utils.data_cache import DataCache


def get_hybrid_model(
    cache: DataCache,
    method: EmbeddingMethod,
    medium: Medium,
    alpha: float,
    beta: float
) -> HybridModel:
    
    """Retrieves hybrid filter model.

    Args:
        item_index (ItemIndex): ItemIndex object of index data.

    Returns:
        HybridModel: Hybrid filter model
    """
    
    content_model = get_content_similarity_model(cache, method, medium)
    item_cf_model = get_item_cf_model(cache, medium)
    user_cf_model = get_user_cf_model(cache, method, medium)
    
    return HybridModel(
        content_model=content_model,
        item_collab_model=item_cf_model,
        user_collab_model=user_cf_model,
        alpha=alpha,
        beta=beta
    )
    
def get_hybrid_embeddings(
    cache: DataCache,
    method: EmbeddingMethod
) -> np.ndarray:
    if method == EmbeddingMethod.SBERT:
        return cache.get("sbert/item_embeddings.npy")
    if method == EmbeddingMethod.TFIDF:
        return cache.get("tfidf/item_embeddings.npy")