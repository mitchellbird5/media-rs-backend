# src/load_data.py
import pandas as pd

from typing import List, Tuple
from src.types.rating import Rating
from src.types.model import IdType

from src.utils.convert_id import (
    get_id_from_value,
)

def load_dataframe(csv_path: str) -> pd.DataFrame:
    """
    Load CSV
    """
    df = pd.read_csv(csv_path).fillna('')
    return df

def get_user_item_matrix(ratings: pd.DataFrame) -> pd.DataFrame:
    return ratings.pivot(index='userId', columns='movieId', values='rating').fillna(0).values

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

def add_user_ratings(
    ratings: pd.DataFrame,
    movies: pd.DataFrame,
    new_ratings: List[Rating],
) -> Tuple[IdType, pd.DataFrame]:
    
    user_id = int(ratings['userId'].max()) + 1
    
    movie_ids = get_id_from_value(
        df=movies,
        keys=[r.title for r in new_ratings],
        search_column='title',
        target_column='movieId'
    )

    
    new_ratings_df = pd.DataFrame({
        'userId': user_id,
        'movieId': movie_ids,
        'rating': [r.rating for r in new_ratings],
    })
    
    return user_id, pd.concat([ratings, new_ratings_df], ignore_index=True)