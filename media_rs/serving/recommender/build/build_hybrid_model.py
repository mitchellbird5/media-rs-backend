from media_rs.serving.recommender.models.hybrid import HybridModel
from media_rs.utils.item_index import ItemIndex
from media_rs.serving.recommender.build.build_collab_model import (
    get_user_cf_model, 
    get_item_cf_model
)
from media_rs.serving.recommender.build.build_content_model import get_content_similarity_model

def get_hybrid_model(
    item_index: ItemIndex,
    alpha: float,
    beta: float
) -> HybridModel:
    
    """Retrieves hybrid filter model.

    Args:
        item_index (ItemIndex): ItemIndex object of index data.

    Returns:
        HybridModel: Hybrid filter model
    """
    
    content_model = get_content_similarity_model(item_index)
    item_cf_model = get_item_cf_model(item_index)
    user_cf_model = get_user_cf_model()
    
    return HybridModel(
        content_model=content_model,
        item_collab_model=item_cf_model,
        user_collab_model=user_cf_model,
        alpha=alpha,
        beta=beta
    )