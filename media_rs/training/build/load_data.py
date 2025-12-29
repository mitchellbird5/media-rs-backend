from pathlib import Path
import pandas as pd
from media_rs.utils.load_data import load_dataframe, add_tags_to_movies

def load_all_data(wdir: Path):
    movies = load_dataframe(wdir.joinpath("movies.csv"))
    ratings = load_dataframe(wdir.joinpath("ratings.csv"))
    tags = load_dataframe(wdir.joinpath("tags.csv"))

    movies = add_tags_to_movies(movies, tags)
    return movies, ratings, tags
