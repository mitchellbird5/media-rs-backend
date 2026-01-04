from media_rs.serving.recommender.models.content import ContentSimilarityModel
from media_rs.utils.movies.movie_data_cache import MovieDataCache

def get_content_similarity_model(cache: MovieDataCache) -> ContentSimilarityModel:
    return ContentSimilarityModel(
        topk_graph=cache.get("movies_item_topk_content.pkl"),
        embeddings=cache.get("movies_item_embeddings.npy"),
        transformer=cache.get("movie_sbert_model")
    )