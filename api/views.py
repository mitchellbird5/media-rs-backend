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
from api.services.database_query import DatabaseService
from media_rs.utils.session import get_or_create_session_id
from media_rs.rs_types.model import EmbeddingMethod, Medium

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

def get_medium(medium: str) -> Medium:
    if medium.lower() == "movies":
        return Medium.MOVIES
    if medium.lower() == "books":
        return Medium.BOOKS
    raise ValueError("Invalid medium")


# -----------------------------
# Endpoints
# -----------------------------
@router.get("/recommend/content", response_model=List[str])
def content_recommendation(
    title: str = Query(...),
    medium: str = Query(...),
    top_n: int = Query(10, ge=1),
    embedding_method: str = Query("SBERT")
):
    method = get_embedding_method(embedding_method)
    medium_enum = get_medium(medium)
    return get_content_recommendations(
        title=title, 
        method=method, 
        top_n=top_n, 
        medium=medium_enum
    )

@router.get("/recommend/content-description", response_model=List[str])
def content_description_recommendation(
    description: str = Query(...),
    medium: str = Query(...),
    top_n: int = Query(10, ge=1),
    embedding_method: str = Query("SBERT")
):
    method = get_embedding_method(embedding_method)
    medium_enum = get_medium(medium)
    return get_content_recommendations_from_description(
        description=description,
        method=method,
        top_n=top_n, 
        medium=medium_enum
    )

@router.get("/recommend/item-cf", response_model=List[str])
def item_cf_recommendation(
    title: str = Query(...),
    medium: str = Query(...),
    top_n: int = Query(10, ge=1)
):
    medium_enum = get_medium(medium)
    return get_item_cf_recommendations(
        title=title, 
        top_n=top_n, 
        medium=medium_enum
    )

@router.post("/recommend/user-cf", response_model=List[str])
def user_cf_recommendation(payload: UserCFInput = Body(...)):
    method = get_embedding_method(payload.embedding_method)
    medium = get_medium(payload.medium)
    try:
        return get_user_cf_recommendations(
            ratings=[r.dict() for r in payload.ratings],
            top_n=payload.top_n,
            k_similar_users=payload.k_similar_users,
            method=method,
            medium=medium
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@router.post("/recommend/hybrid", response_model=List[str])
def hybrid_recommendation(input: HybridInput):
    method = get_embedding_method(input.embedding_method)
    medium = get_medium(input.medium)
    try:
        recs = get_hybrid_recommendations(
            title=input.title,
            ratings=[r.dict() for r in input.ratings],
            alpha=input.alpha,
            beta=input.beta,
            top_n=input.top_n,
            k_similar_users=input.k_similar_users,
            method=method,
            medium=medium
        )
        return recs
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/medium/search")
def medium_search(
    media: str,
    query: str,
    limit: int = Query(10, ge=1, le=100),
):
    """
    Search movies by title.
    Automatically sets a session cookie if missing.
    """
    response = Response()
    
    medium = get_medium(media)

    try:
        # Get session_id from cookie, or create a new one
        sid = get_or_create_session_id(
            response, sid=None
        )

        results = DatabaseService.search_database(
            query=query,
            user_key=f"session:{sid}",
            medium=medium,
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
