import pandas as pd
from typing import Tuple

def build_content_column(movies: pd.DataFrame) -> pd.DataFrame:
    movies = movies.sort_values("movieId").reset_index(drop=True)
    movies["content"] = movies[["genres", "tag"]].fillna("").agg(" ".join, axis=1)
    return movies
