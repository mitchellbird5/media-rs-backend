from media_rs.serving.recommender.models.content import (
    ContentSimilaritySBERTModel, 
    ContentSimilarityTFIDFModel
)
from media_rs.utils.data_cache import DataCache
from media_rs.rs_types.model import EmbeddingMethod, Medium

from typing import Union

def get_content_similarity_tfidf_model(
    cache: DataCache,
    medium: Medium
) -> ContentSimilarityTFIDFModel:
    return ContentSimilarityTFIDFModel(
        topk_graph=cache.get(f"{medium.value}/tfidf/item_topk_content.pkl"),
        embeddings=cache.get(f"{medium.value}/tfidf/item_embeddings.npy"),
        vectorizer=cache.get(f"{medium.value}/tfidf/tfidf_vectorizer.pkl"),
        svd=cache.get(f"{medium.value}/tfidf/svd.pkl")
    )
    
def get_content_similarity_sbert_model(
    cache: DataCache, 
    medium: Medium
) -> ContentSimilaritySBERTModel:
    return ContentSimilaritySBERTModel(
        topk_graph=cache.get(f"{medium.value}/sbert/item_topk_content.pkl"),
        embeddings=cache.get(f"{medium.value}/sbert/item_embeddings.npy"),
        transformer=cache.get(f"{medium.value}/sbert/sbert_model")
    )
    
def get_content_similarity_model(
    cache: DataCache,
    method: EmbeddingMethod,
    medium: Medium  
) -> Union[
    ContentSimilaritySBERTModel, 
    ContentSimilarityTFIDFModel
]:
    if method==EmbeddingMethod.SBERT:
        return get_content_similarity_sbert_model(cache, medium)
    if method==EmbeddingMethod.TFIDF:
        return get_content_similarity_tfidf_model(cache, medium)