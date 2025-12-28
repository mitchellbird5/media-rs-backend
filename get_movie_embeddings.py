import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import faiss

from media_rs.features.faiss import build_faiss_index, query_faiss_topk, save_faiss_index
from media_rs.utils.load_data import (
    load_dataframe,
    add_tags_to_movies,
    save_numpy,
    save_pickle,
    save_metadata
)
from media_rs.features.embeddings import compute_item_embeddings, compute_user_embeddings

# ----------------------------
# Load data
# ----------------------------
movies = load_dataframe("data/movies/raw/movies.csv")
ratings = load_dataframe("data/movies/raw/ratings.csv")
tags = load_dataframe("data/movies/raw/tags.csv")

# Combine tags
movies = add_tags_to_movies(movies, tags)

# ----------------------------
# Stable ordering & index mapping
# ----------------------------
movies = movies.sort_values("movieId").reset_index(drop=True)
movie_ids = movies["movieId"].values
idx_to_movieId = dict(enumerate(movie_ids))
movieId_to_idx = {mid: idx for idx, mid in idx_to_movieId.items()}

# ----------------------------
# Build content features
# ----------------------------
movies["content"] = (
    movies[["genres", "tag"]].fillna("").agg(" ".join, axis=1)
)
movie_texts = movies["content"].values

# ----------------------------
# Compute item embeddings (content)
# ----------------------------
item_embeddings, vectorizer, svd = compute_item_embeddings(movie_texts)

# ----------------------------
# Prepare user-item matrix (for CF)
# ----------------------------
ratings = ratings[ratings["movieId"].isin(movieId_to_idx)]
user_ids = ratings["userId"].unique()
userId_to_idx = {uid: i for i, uid in enumerate(user_ids)}
idx_to_userId = {i: uid for uid, i in userId_to_idx.items()}

num_users = len(user_ids)
num_items = len(movie_ids)

user_item_matrix = np.zeros((num_users, num_items), dtype=np.float32)
for row in ratings.itertuples(index=False):
    u_idx = userId_to_idx[row.userId]
    i_idx = movieId_to_idx[row.movieId]
    user_item_matrix[u_idx, i_idx] = row.rating

# ----------------------------
# Compute user embeddings
# ----------------------------
user_embeddings_dict = compute_user_embeddings(
    {uid: [movieId_to_idx[mid] for mid in ratings[ratings.userId==uid]["movieId"]]
     for uid in user_ids},
    item_embeddings
)

# Convert to NumPy array (FAISS requires 2D float32 array)
user_embeddings = np.vstack([user_embeddings_dict[uid] for uid in user_ids]).astype(np.float32)

# ----------------------------
# Build FAISS index for content similarity
# ----------------------------
faiss_index_content = build_faiss_index(item_embeddings, metric="cosine")
topk_content = query_faiss_topk(faiss_index_content, item_embeddings, k=100)
save_faiss_index(faiss_index_content, "data/movies/faiss_index_content.index")

# ----------------------------
# Build item-item CF top-K graph
# ----------------------------
item_sim_matrix = cosine_similarity(user_item_matrix.T)
topk_cf = {}
k = 100
for i in range(num_items):
    top_indices = np.argsort(-item_sim_matrix[i])[:k+1]
    topk_cf[i] = [(j, item_sim_matrix[i, j]) for j in top_indices if j != i]

# ----------------------------
# Build user-user CF FAISS index
# ----------------------------
faiss.normalize_L2(user_embeddings)
faiss_index_users = faiss.IndexFlatIP(user_embeddings.shape[1])  # cosine similarity
faiss_index_users.add(user_embeddings)
save_faiss_index(faiss_index_users, "data/movies/users_faiss.index")

# ----------------------------
# Save all artifacts
# ----------------------------
save_pickle(idx_to_movieId, "data/movies/idx_to_movieId.pkl")
save_pickle(movieId_to_idx, "data/movies/movieId_to_idx.pkl")
save_pickle(idx_to_userId, "data/movies/users_idx_to_userId.pkl")
save_pickle(userId_to_idx, "data/movies/users_userId_to_idx.pkl")  # <-- saved for FAISS lookup

save_numpy(item_embeddings, "data/movies/movies_item_embeddings.npy")
save_numpy(user_embeddings, "data/movies/movies_user_embeddings.npy")

save_pickle(topk_content, "data/movies/movies_item_topk_content.pkl")
save_pickle(topk_cf, "data/movies/movies_item_topk_cf.pkl")
save_numpy(user_item_matrix, "data/movies/user_item_matrix.npy")

save_pickle(vectorizer, "data/movies/movies_vectorizer.pkl")
save_pickle(svd, "data/movies/movies_svd.pkl")

save_metadata(movies, "data/movies/movies_item_metadata.parquet")
