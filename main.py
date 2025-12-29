# main.py
import numpy as np
import pickle
import faiss
from pathlib import Path

from scipy.sparse import load_npz

from media_rs.utils.load_data import (
    load_dataframe,
    add_tags_to_movies,
    get_user_item_matrix,
    get_user_ratings
)
from media_rs.utils.convert_id import (
    get_id_from_value,
    get_value_from_id,
    get_result_from_similarity
)

from media_rs.models.content import ContentModel
from media_rs.models.collab import ItemItemCollaborativeModel, UserCollaborativeModel
from media_rs.models.hybrid import HybridModel

from media_rs.rs_types.rating import Rating

wdir = Path("data/movies/raw/ml-latest/")

movies_csv_path = wdir.joinpath("movies.csv")

movieId_to_idx_path = wdir.joinpath("movieId_to_idx.pkl")
idx_to_movieId_path = wdir.joinpath("idx_to_movieId.pkl")
userId_to_idx_path = wdir.joinpath("users_userId_to_idx.pkl")
idx_to_userId_path = wdir.joinpath("users_idx_to_userId.pkl")

item_content_topk_path = wdir.joinpath("movies_item_topk_content.pkl")
item_cf_topk_path = wdir.joinpath("movies_item_topk_cf.pkl")
user_item_matrix_path = wdir.joinpath("user_item_matrix.npz")

item_embeddings_path = wdir.joinpath("movies_item_embeddings.npy")

vectorizer_path = wdir.joinpath("movies_vectorizer.pkl")
svd_path = wdir.joinpath("movies_svd.pkl")

faiss_index_path = str(wdir.joinpath("users_faiss.index"))

# Step 1: Load movie data
movies = load_dataframe(movies_csv_path)

movieId_to_idx = pickle.load(open(movieId_to_idx_path, "rb"))
idx_to_movieId = pickle.load(open(idx_to_movieId_path, "rb"))
userId_to_idx = pickle.load(open(userId_to_idx_path, "rb"))
idx_to_userId = pickle.load(open(idx_to_userId_path, "rb"))

with open(item_content_topk_path, "rb") as f:
    topk_graph_content = pickle.load(f)
with open(item_cf_topk_path, "rb") as f:
    topk_graph_cf = pickle.load(f)
user_item_matrix = load_npz(user_item_matrix_path)
    
item_embeddings = np.load(item_embeddings_path)

vectorizer = pickle.load(open(vectorizer_path, "rb"))
svd = pickle.load(open(svd_path, "rb"))

faiss_index = faiss.read_index(faiss_index_path)


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
new_ratings = get_user_ratings(movies, user_ratings, movieId_to_idx=movieId_to_idx)

movie_indices = movies["movieId"].map(movieId_to_idx).to_numpy()

movie_id = get_id_from_value(
    df=movies,
    keys=[movie_name], 
    search_column='title', 
    target_column='movieId'
)[0]

item_idx = movieId_to_idx[movie_id]
user_idx = userId_to_idx[1]


rs_content = ContentModel(
    ids=movie_indices,
    topk_graph=topk_graph_content,
    embeddings=item_embeddings,
    vectorizer=vectorizer,
    svd=svd
)
content_similarity = rs_content.recommend(item_idx, 10)
content_movies = get_result_from_similarity(
    df=movies,
    result=content_similarity
)
content_text = rs_content.recommend_from_text(
    text="crime thriller"
)
content_text_movies = get_result_from_similarity(
    df=movies,
    result=content_text
)


rs_item = ItemItemCollaborativeModel(
    ids=movie_indices,
    topk_graph=topk_graph_cf
)
item_similarity = rs_item.recommend(item_idx, 10)
item_movies = get_result_from_similarity(
    df=movies,
    result=item_similarity
)


rs_user = UserCollaborativeModel(
    faiss_index=faiss_index,
    user_item_matrix=user_item_matrix
)
user_similarity = rs_user.recommend(
    ratings=new_ratings,
    item_embeddings=item_embeddings
)
user_movies = get_result_from_similarity(
    df=movies,
    result=user_similarity
)


rs_hybrid = HybridModel(
    content_model=rs_content,
    item_collab_model=rs_item,
    user_collab_model=rs_user
)
hybrid_similarity = rs_hybrid.recommend(
    item_idx=item_idx,
    new_user_ratings=new_ratings,
    item_embeddings=item_embeddings
)
hybrid_movies = get_result_from_similarity(
    df=movies,
    result=hybrid_similarity
)


print()