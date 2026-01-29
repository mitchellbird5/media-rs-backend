# views.py
from fastapi import APIRouter, HTTPException, Query, Request, Response, Body
from typing import List
from api.services.content_services import (
    get_content_recommendations,
    get_content_recommendations_from_description,
)
from api.services.collab_services import (
    get_item_cf_recommendations,
    get_user_cf_recommendations
)
from api.services.hybrid_services import get_hybrid_recommendations
from api.services.tmdb import get_multiple_movie_data
from api.services.database_query import MoviesService
from media_rs.utils.session import get_or_create_session_id
from media_rs.rs_types.model import EmbeddingMethod

from .serializers import (
    ContentRecommendationInput,
    ContentDescriptionInput,
    UserCFInput,
    HybridInput,
    MovieSearchInput,
    RecommendationListOutput
)

# -----------------------------
# Router
# -----------------------------
router = APIRouter(
    prefix="/api",
    tags=["recommendations"]
)

# -----------------------------
# Helper
# -----------------------------
def get_embedding_method(method: str) -> EmbeddingMethod:
    if method.upper() == "SBERT":
        return EmbeddingMethod.SBERT
    if method.upper() == "TFIDF":
        return EmbeddingMethod.TFIDF
    raise ValueError("Invalid embedding method")

# -----------------------------
# Endpoints
# -----------------------------
@router.get("/recommend/content", response_model=List[str])
def content_recommendation(
    movie_title: str = Query(...),
    top_n: int = Query(10, ge=1),
    embedding_method: str = Query("SBERT")
):
    method = get_embedding_method(embedding_method)
    return get_content_recommendations(movie_title, method, top_n)

@router.get("/recommend/content-description", response_model=List[str])
def content_description_recommendation(
    description: str = Query(...),
    top_n: int = Query(10, ge=1),
    embedding_method: str = Query("SBERT")
):
    method = get_embedding_method(embedding_method)
    return get_content_recommendations_from_description(description, method, top_n)

@router.get("/recommend/item-cf", response_model=List[str])
def item_cf_recommendation(
    movie_title: str = Query(...),
    top_n: int = Query(10, ge=1)
):
    return get_item_cf_recommendations(movie_title, top_n)

@router.post("/recommend/user-cf", response_model=List[str])
def user_cf_recommendation(payload: UserCFInput = Body(...)):
    method = get_embedding_method(payload.embedding_method)
    try:
        return get_user_cf_recommendations(
            ratings=[r.dict() for r in payload.ratings],
            top_n=payload.top_n,
            k_similar_users=payload.k_similar_users,
            method=method
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@router.post("/recommend/hybrid", response_model=List[str])
def hybrid_recommendation(input: HybridInput):
    method = get_embedding_method(input.embedding_method)
    try:
        recs = get_hybrid_recommendations(
            movie_title=input.movie_title,
            ratings=[r.dict() for r in input.ratings],
            alpha=input.alpha,
            beta=input.beta,
            top_n=input.top_n,
            k_similar_users=input.k_similar_users,
            method=method
        )
        return recs
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/movies/search")
def movie_search(
    query: str,
    limit: int = Query(10, ge=1, le=100),
):
    """
    Search movies by title.
    Automatically sets a session cookie if missing.
    """
    response = Response()

    try:
        # Get session_id from cookie, or create a new one
        sid = get_or_create_session_id(
            response, sid=None
        )

        results = MoviesService.search_movies(
            query=query,
            user_key=f"session:{sid}",
            limit=limit
        )

        # Return results as JSON with the cookie set
        response.media_type = "application/json"
        response.body = results.json().encode() if hasattr(results, "json") else str(results).encode()
        return results

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/movies/data")
def movie_data(titles: List[str] = Query(...)):
    if not titles:
        raise HTTPException(status_code=400, detail="At least one title required")
    try:
        data = get_multiple_movie_data(titles)
        return [d.__dict__ for d in data]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
