import pandas as pd
import numpy as np
import pickle

from pathlib import Path
from typing import Tuple

def load_dataframe(csv_path: str) -> pd.DataFrame:
    """
    Load CSV
    """
    df = pd.read_csv(csv_path).fillna('')
    return df

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

def load_all_data(
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

def save_metadata(df: pd.DataFrame, path: str):
    df.to_parquet(path, index=False)

def load_metadata(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)
