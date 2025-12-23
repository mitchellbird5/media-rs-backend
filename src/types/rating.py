from pydantic import BaseModel, Field

class Rating(BaseModel):
    title: str
    rating: float = Field(ge=0, le=5)