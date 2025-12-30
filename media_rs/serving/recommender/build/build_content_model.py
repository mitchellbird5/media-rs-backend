import numpy as np
import pickle

from pathlib import Path
from sentence_transformers import SentenceTransformer

from media_rs.serving.recommender.models.content import ContentSimilarityModel
from sentence_transformers import SentenceTransformer

def get_content_similarity_model() -> ContentSimilarityModel:
    """Retrieves content filter model.

    Returns:
        ContentSimilarityModel: Content filter model
    """
    
    wdir = Path("media_rs/serving/artifacts")

    # Preload metadata for faster API responses
    item_content_topk_path = wdir.joinpath("movies_item_topk_content.pkl")
    transformer_path = wdir.joinpath("movie_sbert_model")
    item_embeddings_path = wdir.joinpath("movies_item_embeddings.npy")

    # Load content model artifacts
    with open(item_content_topk_path, "rb") as f:
        topk_graph_content = pickle.load(f)
    item_embeddings = np.load(item_embeddings_path)
    transformer = SentenceTransformer(str(transformer_path))

    # Initialize content model
    return ContentSimilarityModel(
        topk_graph=topk_graph_content,
        embeddings=item_embeddings,
        transformer=transformer
    )