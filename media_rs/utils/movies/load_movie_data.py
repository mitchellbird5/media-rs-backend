import pandas as pd

from pathlib import Path
from typing import Tuple

from media_rs.utils.load_data import (
    load_dataframe
)

def load_all_movie_data(
    wdir: Path
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Loads movie data to train model

    Args:
        wdir (Path): Directory containing data

    Returns:
        _type_: _description_
    """
    
    movies = load_dataframe(wdir.joinpath("movies.csv"))
    ratings = load_dataframe(wdir.joinpath("ratings.csv"))
    tags = load_dataframe(wdir.joinpath("tags.csv"))

    movies = add_tags_to_movies(movies, tags)
    return movies, ratings, tags

def add_tags_to_movies(
    movies: pd.DataFrame, 
    tags: pd.DataFrame
) -> pd.DataFrame:
    # Group tags by movieId
    movie_tags = tags.groupby('movieId')['tag'].apply(lambda x: ' '.join(x)).reset_index()
    # Merge with movies
    movies = movies.merge(movie_tags, on='movieId', how='left')
    movies['tag'] = movies['tag'].fillna('')
    return movies

