# src/models/base.py
import pandas as pd
from abc import ABC, abstractmethod
from typing import List


class BaseRecommender(ABC):
    def __init__(self, movies: pd.DataFrame):
        self.movies = movies

    def _movies_from_ids(self, movie_ids: List[int]) -> pd.DataFrame:
        return self.movies[self.movies["movieId"].isin(movie_ids)][
            ["title", "genres"]
        ]

    @abstractmethod
    def recommend(self, *args, **kwargs) -> pd.DataFrame:
        pass
