# main.py
from src.utils.load_data import (
    load_dataframe,
    add_tags_to_movies,
    get_user_item_matrix
)
from src.utils.concatenate_features import concat_string_columns
from src.utils.convert_id import (
    get_id_from_value,
    get_value_from_id,
    get_result_from_similarity
)

from src.models.content import ContentModel
from src.models.collab import CollaborativeModel
from src.models.hybrid import HybridModel
from src.types.rating import Rating

from src.types.model import CollabMethod

# Step 1: Load movie data
movies = load_dataframe('data/raw/ml-latest-small/movies.csv')
ratings = load_dataframe('data/raw/ml-latest-small/ratings.csv')
tags = load_dataframe('data/raw/ml-latest-small/tags.csv')

movies = add_tags_to_movies(movies, tags)

user_item_matrix = get_user_item_matrix(ratings)

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

rs_content = ContentModel(ids, features)
content_similarity = rs_content.recommend(movie_id, n)
content_movies = get_result_from_similarity(
    df=movies,
    keys=tags_to_include,
    search_column='movieId',
    result=content_similarity
)

rs_collab_item = CollaborativeModel(ids, user_item_matrix, CollabMethod.ITEM)
item_similarity = rs_collab_item.recommend(movie_id, n)
item_movies = get_result_from_similarity(
    df=movies,
    keys=tags_to_include,
    search_column='movieId',
    result=item_similarity
)

rs_collab_user = CollaborativeModel(ids, user_item_matrix, CollabMethod.USER)
user_similarity = rs_collab_user.recommend(movie_id, n)
user_movies = get_result_from_similarity(
    df=movies,
    keys=tags_to_include,
    search_column='movieId',
    result=user_similarity
)

rs_hybrid = HybridModel(ids, rs_content, rs_collab_user)
hybrid_ids = rs_hybrid.recommend(movie_id, n)
hybrid_movies = get_value_from_id(
    movies,
    hybrid_ids,
    search_column="movieId"
)

print()