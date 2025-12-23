# src/models/base.py
import pandas as pd
from abc import ABC, abstractmethod
from typing import List


class BaseRecommender(ABC):
    def __init__(self, movies: pd.DataFrame):
        self.movies = movies

    @abstractmethod
    def recommend(self, *args, **kwargs) -> pd.DataFrame:
        pass
