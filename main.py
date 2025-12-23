# main.py
import pandas as pd

from typing import Optional

from src.utils.load_data import (
    load_dataframe,
    add_tags_to_movies
)
from src.utils.concatenate_features import concat_string_columns
from src.utils.convert_id import (
    get_id_from_value,
    get_result_from_similarity
)

from src.recommender import RecommenderSystem
from src.types.rating import Rating

# Step 1: Load movie data
movies = load_dataframe('data/raw/ml-latest-small/movies.csv')
ratings = load_dataframe('data/raw/ml-latest-small/ratings.csv')
tags = load_dataframe('data/raw/ml-latest-small/tags.csv')

movies = add_tags_to_movies(movies, tags)

movie_name = "Toy Story (1995)"
tags_to_include=[
    'title',
    'genres',
    'tag'
]
n = 10

user_ratings = [
    Rating(
        title="Toy Story (1995)",
        rating=5
    ),
    Rating(
        title="Nixon (1995)",
        rating=1
    ),
    Rating(
        title="Beauty and the Beast (1991)",
        rating=4
    ),
    Rating(
        title="Cool Runnings (1993)",
        rating=4
    ),
]

ids = movies['movieId'].values
features = concat_string_columns(movies, tags_to_include)
movie_id = get_id_from_value(
    df=movies,
    keys=[movie_name], 
    search_column='title', 
    target_column='movieId'
)

rs = RecommenderSystem(ids, features)

content_similarity = rs.content.recommend(movie_id, n)
content_movies = get_result_from_similarity(
    df=movies,
    keys=tags_to_include,
    search_column='movieId',
    result=content_similarity
)

# print(rs.collaborative_item.recommend(movie_name, n), movie_name, n)
# print(rs.collaborative_user.recommend(user_ratings, n), movie_name, n)
# print(rs.hybrid.recommend(movie_name, n), movie_name, n)

print()