# src/load_data.py
import pandas as pd
import numpy as np
import pickle

from typing import List, Tuple
from media_rs.rs_types.rating import Rating
from media_rs.rs_types.model import IdType

from media_rs.utils.convert_id import (
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

# ----------------------------
# 4. Save and Load Utilities
# ----------------------------
def save_numpy(array: np.ndarray, path: str):
    np.save(path, array)

def load_numpy(path: str) -> np.ndarray:
    return np.load(path)

def save_pickle(obj, path: str):
    with open(path, "wb") as f:
        pickle.dump(obj, f)

def load_pickle(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)

# ----------------------------
# 5. Generic Metadata Save
# ----------------------------
def save_metadata(df: pd.DataFrame, path: str):
    df.to_parquet(path, index=False)

def load_metadata(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)
