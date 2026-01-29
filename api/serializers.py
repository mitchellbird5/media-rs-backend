# serializers.py
from typing import List, Union
from pydantic import BaseModel, Field

Embedding = Union['SBERT', 'TFIDF']

# -----------------------------
# Input models
# -----------------------------
class Rating(BaseModel):
    name: str
    value: float = Field(..., ge=0, le=5)

class ContentRecommendationInput(BaseModel):
    movie_title: str
    top_n: int = 10
    embedding_method: Embedding = "SBERT"

class ContentDescriptionInput(BaseModel):
    description: str
    top_n: int = 10
    embedding_method: Embedding = "SBERT"
    
class ItemItemCFInput(BaseModel):
    movie_title: str
    top_n: int = 10

class UserCFInput(BaseModel):
    ratings: List[Rating]
    top_n: int = 10
    k_similar_users: int = 50
    embedding_method: Embedding = "SBERT"

class HybridInput(BaseModel):
    movie_title: str
    alpha: float = Field(0.5, ge=0, le=1)
    beta: float = Field(0.3, ge=0, le=1)
    ratings: List[Rating]
    top_n: int = 10
    k_similar_users: int = 50
    embedding_method: Embedding = "SBERT"

class MovieSearchInput(BaseModel):
    query: str
    limit: int = Field(10, ge=1, le=100)

# -----------------------------
# Output models
# -----------------------------
class ContentRecommendationOutput(BaseModel):
    title: str

class RecommendationListOutput(BaseModel):
    recommendations: List[ContentRecommendationOutput]
