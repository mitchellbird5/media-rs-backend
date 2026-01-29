# views.py
from fastapi import APIRouter, HTTPException, Query, Request, Response
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
    RecommendationListOutput,
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
@router.get("/recommend/content", response_model=RecommendationListOutput)
def content_recommendation(
    movie_title: str = Query(...),
    top_n: int = Query(10, ge=1),
    embedding_method: str = Query("SBERT")
):
    method = get_embedding_method(embedding_method)
    recs = get_content_recommendations(movie_title, method, top_n)
    return {"recommendations": recs}

@router.get("/recommend/content-description", response_model=RecommendationListOutput)
def content_description_recommendation(
    description: str = Query(...),
    top_n: int = Query(10, ge=1),
    embedding_method: str = Query("SBERT")
):
    method = get_embedding_method(embedding_method)
    recs = get_content_recommendations_from_description(description, method, top_n)
    return {"recommendations": recs}

@router.get("/recommend/item-cf", response_model=RecommendationListOutput)
def item_cf_recommendation(
    movie_title: str = Query(...),
    top_n: int = Query(10, ge=1)
):
    recs = get_item_cf_recommendations(movie_title, top_n)
    return {"recommendations": recs}

@router.post("/recommend/user-cf", response_model=RecommendationListOutput)
def user_cf_recommendation(input: UserCFInput):
    method = get_embedding_method(input.embedding_method)
    try:
        recs = get_user_cf_recommendations(
            ratings=[r.dict() for r in input.ratings],
            top_n=input.top_n,
            k_similar_users=input.k_similar_users,
            method=method
        )
        return {"recommendations": recs}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/recommend/hybrid", response_model=RecommendationListOutput)
def hybrid_recommendation(input: HybridInput):
    if input.alpha + input.beta > 1:
        raise HTTPException(status_code=400, detail="alpha + beta must be <= 1")
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
        return {"recommendations": recs}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/movies/search")
def movie_search(
    request: Request,
    response: Response,
    query: str = Query(...),
    limit: int = Query(10, ge=1, le=100)
):
    try:
        session_id = get_or_create_session_id(request, response)
        results = MoviesService.search_movies(
            query=query,
            user_key=f"session:{session_id}",
            limit=limit
        )
        return {"results": results}
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
