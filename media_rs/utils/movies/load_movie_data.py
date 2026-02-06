import pandas as pd
import re

from pathlib import Path
from typing import Tuple

from media_rs.utils.load_data import (
    load_dataframe
)

def remove_year_from_title(title: str) -> str:
    """
    Removes a year in parentheses from the end of a movie title.
    Example: "Toy Story (1995)" -> "Toy Story"

    Args:
        title (str): Original movie title

    Returns:
        str: Title without year
    """
    if not isinstance(title, str):
        return title

    return re.sub(r'\s*\(\d{4}\)$', '', title).strip()

def add_title_no_year_column(movies: pd.DataFrame, column: str = "title") -> pd.DataFrame:
    """
    Adds a new column 'title_no_year' to the movies DataFrame
    with the year removed from titles.

    Args:
        movies (pd.DataFrame): Movies dataframe
        column (str): Column containing the title with year

    Returns:
        pd.DataFrame: Movies dataframe with 'title_no_year' column added
    """
    movies = movies.copy()
    movies['title_no_year'] = movies[column].apply(remove_year_from_title)
    return movies

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
    
    movies = movies.rename(columns={"movieId" : "itemId"})
    ratings = ratings.rename(columns={"movieId" : "itemId"})
    tags = tags.rename(columns={"movieId" : "itemId"})
    links = links.rename(columns={"movieId" : "itemId"})

    movies = add_tags_to_movies(movies, tags)
    movies = add_title_no_year_column(movies)
    movies["title_norm"] = movies["title_no_year"]
    
    return movies, ratings, tags, links

def add_tags_to_movies(
    movies: pd.DataFrame, 
    tags: pd.DataFrame
) -> pd.DataFrame:
    # Group tags by itemId
    movie_tags = tags.groupby('itemId')['tag'].apply(lambda x: ' '.join(x)).reset_index()
    # Merge with movies
    movies = movies.merge(movie_tags, on='itemId', how='left')
    movies['tag'] = movies['tag'].fillna('')
    return movies

