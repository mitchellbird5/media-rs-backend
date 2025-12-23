# src/load_data.py
import pandas as pd

from typing import Sequence, List

def load_dataframe(csv_path: str) -> pd.DataFrame:
    """
    Load CSV
    """
    df = pd.read_csv(csv_path).fillna('')
    return df

def get_user_item_matrix(ratings: pd.DataFrame) -> pd.DataFrame:
    return ratings.pivot(index='userId', columns='movieId', values='rating').fillna(0)

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