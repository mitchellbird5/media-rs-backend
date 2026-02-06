# serializers.py
from typing import List, Literal
from pydantic import BaseModel, Field, model_validator

Embedding = Literal["SBERT", "TFIDF"]
Medium = Literal["movies", "books"]
# -----------------------------
# Input models
# -----------------------------
class Rating(BaseModel):
    name: str
    value: float = Field(..., ge=0, le=5)

class ContentRecommendationInput(BaseModel):
    title: str
    top_n: int = 10
    embedding_method: Embedding = "SBERT"

class ContentDescriptionInput(BaseModel):
    description: str
    top_n: int = 10
    embedding_method: Embedding = "SBERT"
    
class ItemItemCFInput(BaseModel):
    title: str
    top_n: int = 10

class UserCFInput(BaseModel):
    ratings: List[Rating]
    medium: Medium
    top_n: int = Field(10, ge=1)
    k_similar_users: int = Field(50, ge=1)
    embedding_method: Embedding = "SBERT"

class HybridInput(BaseModel):
    title: str
    medium: Medium
    alpha: float = Field(0.5, ge=0, le=1)
    beta: float = Field(0.3, ge=0, le=1)
    ratings: List[Rating]
    top_n: int = Field(10, ge=1)
    k_similar_users: int = Field(50, ge=1)
    embedding_method: Embedding = "SBERT"

    @model_validator(mode="after")
    def check_alpha_beta(cls, model):
        if model.alpha + model.beta > 1:
            raise ValueError("alpha + beta must be <= 1")
        return model

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
