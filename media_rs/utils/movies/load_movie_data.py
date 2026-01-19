import pandas as pd
import re
import unicodedata

from pathlib import Path
from typing import Tuple

from media_rs.utils.load_data import (
    load_dataframe
)

def norm(title: str) -> str:
    if not title:
        return ""

    # Unicode normalization (é → é)
    title = unicodedata.normalize("NFKC", title)

    # Lowercase
    title = title.lower()

    # Strip surrounding whitespace
    title = title.strip()

    # Replace common punctuation with spaces
    title = re.sub(r"[^\w\s()\-]", " ", title)

    # Collapse whitespace
    title = re.sub(r"\s+", " ", title)

    return title

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
    links = load_dataframe(wdir.joinpath("links.csv"))

    movies = add_tags_to_movies(movies, tags)
    movies["title_norm"] = movies["title"].apply(norm)
    
    return movies, ratings, tags, links

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

