import pandas as pd
from media_rs.utils.movies.load_movie_data import norm

def build_content_column(movies: pd.DataFrame) -> pd.DataFrame:
    """
    Builds column of text content to use in embedding

    Args:
        movies (pd.DataFrame): Dataframe of movie data

    Returns:
        pd.DataFrame: Updated dataframe with content column
    """
    movies = movies.sort_values("movieId").reset_index(drop=True)
    movies["content"] = movies[["title", "genres", "tag"]].fillna("").agg(" ".join, axis=1).apply(norm)
    return movies
