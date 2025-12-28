# main.py
from media_rs.utils.load_data import (
    load_dataframe,
    add_tags_to_movies,
    get_user_item_matrix,
    add_user_ratings
)
from media_rs.utils.concatenate_features import concat_string_columns
from media_rs.utils.convert_id import (
    get_id_from_value,
    get_value_from_id,
    get_result_from_similarity
)

from media_rs.models.content import ContentModel
from media_rs.models.collab import CollaborativeModel
from media_rs.models.hybrid import HybridModel
from media_rs.rs_types.rating import Rating

from media_rs.rs_types.model import CollabMethod

# Step 1: Load movie data
movies = load_dataframe('data/raw/ml-latest-small/movies.csv')
ratings = load_dataframe('data/raw/ml-latest-small/ratings.csv')
tags = load_dataframe('data/raw/ml-latest-small/tags.csv')

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
        title="Antz (1998)",
        rating=5
    ),
    Rating(
        title="Toy Story 2 (1999)",
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

movies = add_tags_to_movies(movies, tags)
user_id, ratings = add_user_ratings(ratings, movies, user_ratings)

user_item_matrix = get_user_item_matrix(ratings)
movie_ids = movies['movieId'].unique()
user_ids = ratings['userId'].unique()

features = concat_string_columns(movies, tags_to_include)
movie_id = get_id_from_value(
    df=movies,
    keys=[movie_name], 
    search_column='title', 
    target_column='movieId'
)[0]

rs_content = ContentModel(movie_ids, features)
content_similarity = rs_content.recommend(movie_id, n)
content_movies = get_result_from_similarity(
    df=movies,
    keys=tags_to_include,
    search_column='movieId',
    result=content_similarity
)

rs_collab_item = CollaborativeModel(movie_ids, user_item_matrix.T, CollabMethod.ITEM)
item_similarity = rs_collab_item.recommend(movie_id, n)
item_movies = get_result_from_similarity(
    df=movies,
    keys=tags_to_include,
    search_column='movieId',
    result=item_similarity
)

rs_collab_user = CollaborativeModel(user_ids, user_item_matrix, CollabMethod.USER)
user_similarity = rs_collab_user.recommend(user_id, n)
user_movies = get_result_from_similarity(
    df=movies,
    keys=tags_to_include,
    search_column='movieId',
    result=user_similarity
)

rs_hybrid = HybridModel(movie_ids, rs_content, rs_collab_user)
hybrid_ids = rs_hybrid.recommend(movie_id, n)
hybrid_movies = get_value_from_id(
    movies,
    hybrid_ids,
    search_column="movieId"
)

print()