import numpy as np
import faiss
from pathlib import Path
from scipy.sparse import coo_matrix, save_npz
from sklearn.metrics.pairwise import cosine_similarity

from media_rs.features.faiss import build_faiss_index, query_faiss_topk, save_faiss_index
from media_rs.utils.load_data import (
    load_dataframe,
    add_tags_to_movies,
    save_numpy,
    save_pickle,
    save_metadata
)
from media_rs.features.embeddings import compute_item_embeddings, compute_user_embeddings

wdir = Path("data/movies/raw/ml-latest/")

# ----------------------------
# Load data
# ----------------------------
movies = load_dataframe(wdir.joinpath("movies.csv"))
ratings = load_dataframe(wdir.joinpath("ratings.csv"))
tags = load_dataframe(wdir.joinpath("tags.csv"))

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
# Prepare sparse user-item matrix
# ----------------------------
ratings = ratings[ratings["movieId"].isin(movieId_to_idx)]
user_ids = ratings["userId"].unique()
userId_to_idx = {uid: i for i, uid in enumerate(user_ids)}
idx_to_userId = {i: uid for uid, i in userId_to_idx.items()}

rows = [userId_to_idx[r.userId] for r in ratings.itertuples(index=False)]
cols = [movieId_to_idx[r.movieId] for r in ratings.itertuples(index=False)]
data = [r.rating for r in ratings.itertuples(index=False)]

user_item_matrix = coo_matrix((data, (rows, cols)), shape=(len(user_ids), len(movie_ids)), dtype=np.float32).tocsr()

# ----------------------------
# Compute user embeddings (sparse-aware)
# ----------------------------
user_embeddings_dict = compute_user_embeddings(
    {uid: user_item_matrix[userId_to_idx[uid]].indices.tolist() for uid in user_ids},
    item_embeddings
)

user_embeddings = np.vstack([user_embeddings_dict[uid] for uid in user_ids]).astype(np.float32)

# ----------------------------
# Build FAISS index for content similarity
# ----------------------------
faiss_index_content = build_faiss_index(item_embeddings, metric="cosine")
topk_content = query_faiss_topk(faiss_index_content, item_embeddings, k=100)

# ----------------------------
# Build item-item CF top-K using sparse matrix
# ----------------------------
# We'll compute similarity in batches to avoid memory blowup
num_items = len(movie_ids)
k = 100
topk_cf = {}

batch_size = 1000
for start in range(0, num_items, batch_size):
    end = min(start + batch_size, num_items)
    batch_matrix = user_item_matrix[:, start:end].T  # items x users
    sim = cosine_similarity(batch_matrix, user_item_matrix.T)
    for i, item_idx in enumerate(range(start, end)):
        top_indices = np.argsort(-sim[i])[:k+1]
        topk_cf[item_idx] = [(j, sim[i, j]) for j in top_indices if j != item_idx]

# ----------------------------
# Build user-user CF FAISS index
# ----------------------------
faiss.normalize_L2(user_embeddings)
faiss_index_users = faiss.IndexFlatIP(user_embeddings.shape[1])
faiss_index_users.add(user_embeddings)

# ----------------------------
# Save all artifacts
# ----------------------------
save_faiss_index(faiss_index_content, str(wdir.joinpath("faiss_index_content.index")))
save_faiss_index(faiss_index_users, str(wdir.joinpath("users_faiss.index")))

save_pickle(idx_to_movieId, wdir.joinpath("idx_to_movieId.pkl"))
save_pickle(movieId_to_idx, wdir.joinpath("movieId_to_idx.pkl"))
save_pickle(idx_to_userId, wdir.joinpath("users_idx_to_userId.pkl"))
save_pickle(userId_to_idx, wdir.joinpath("users_userId_to_idx.pkl"))

save_numpy(item_embeddings, wdir.joinpath("movies_item_embeddings.npy"))
save_numpy(user_embeddings, wdir.joinpath("movies_user_embeddings.npy"))

save_pickle(topk_content, wdir.joinpath("movies_item_topk_content.pkl"))
save_pickle(topk_cf, wdir.joinpath("movies_item_topk_cf.pkl"))
save_npz(wdir.joinpath("user_item_matrix.npz"), user_item_matrix)

save_pickle(vectorizer, wdir.joinpath("movies_vectorizer.pkl"))
save_pickle(svd, wdir.joinpath("movies_svd.pkl"))

save_metadata(movies, wdir.joinpath("movies_item_metadata.parquet"))
