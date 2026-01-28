from media_rs.serving.recommender.models.content import (
    ContentSimilaritySBERTModel, 
    ContentSimilarityTFIDFModel
)
from media_rs.utils.movies.movie_data_cache import MovieDataCache
from media_rs.rs_types.model import EmbeddingMethod

from typing import Union

def get_content_similarity_tfidf_model(cache: MovieDataCache) -> ContentSimilarityTFIDFModel:
    return ContentSimilarityTFIDFModel(
        topk_graph=cache.get("tfidf/item_topk_content.pkl"),
        embeddings=cache.get("tfidf/item_embeddings.npy"),
        vectorizer=cache.get("tfidf/tfidf_vectorizer.pkl"),
        svd=cache.get("tfidf/svd.pkl")
    )
    
def get_content_similarity_sbert_model(cache: MovieDataCache) -> ContentSimilaritySBERTModel:
    return ContentSimilaritySBERTModel(
        topk_graph=cache.get("sbert/item_topk_content.pkl"),
        embeddings=cache.get("sbert/item_embeddings.npy"),
        transformer=cache.get("sbert/sbert_model")
    )
    
def get_content_similarity_model(
    cache: MovieDataCache,
    method: EmbeddingMethod
) -> Union[
    ContentSimilaritySBERTModel, 
    ContentSimilarityTFIDFModel
]:
    if method==EmbeddingMethod.SBERT:
        return get_content_similarity_sbert_model(cache)
    if method==EmbeddingMethod.TFIDF:
        return get_content_similarity_tfidf_model(cache)