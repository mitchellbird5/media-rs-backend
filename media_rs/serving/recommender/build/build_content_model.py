import numpy as np
import pickle

from pathlib import Path

from media_rs.serving.recommender.models.content import ContentModel
from media_rs.utils.item_index import ItemIndex

def get_content_model() -> ContentModel:
    """Retrieves content filter model.

    Returns:
        ContentModel: Content filter model
    """
    
    wdir = Path("media_rs/serving/artifacts")

    # Preload metadata for faster API responses
    item_content_topk_path = wdir.joinpath("movies_item_topk_content.pkl")
    item_embeddings_path = wdir.joinpath("movies_item_embeddings.npy")
    vectorizer_path = wdir.joinpath("movies_vectorizer.pkl")
    svd_path = wdir.joinpath("movies_svd.pkl")

    # Load content model artifacts
    with open(item_content_topk_path, "rb") as f:
        topk_graph_content = pickle.load(f)
    item_embeddings = np.load(item_embeddings_path)
    vectorizer = pickle.load(open(vectorizer_path, "rb"))
    svd = pickle.load(open(svd_path, "rb"))

    # Initialize content model
    return ContentModel(
        topk_graph=topk_graph_content,
        embeddings=item_embeddings,
        vectorizer=vectorizer,
        svd=svd
    )