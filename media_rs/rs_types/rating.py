from pydantic import BaseModel, Field
from typing import TypedDict

class Rating(TypedDict):
    name: str
    value: float