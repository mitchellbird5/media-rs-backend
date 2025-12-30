from pathlib import Path
from scipy.sparse import save_npz

from media_rs.training.build.load_data import load_all_data
from media_rs.training.build.build_content_features import build_content_column
from media_rs.training.build.build_user_item_matrix import build_user_item_matrix
from media_rs.training.build.compute_embeddings import compute_item_and_user_embeddings
from media_rs.training.build.build_topk_graphs import build_item_cf_topk, build_topk_content
from media_rs.training.build.build_faiss_indices import build_faiss_indices
from media_rs.training.build.load_data import save_pickle, save_numpy
from media_rs.training.features.faiss import save_faiss_index

save_dir = Path("media_rs/serving/artifacts/")
file_dir = Path("data/movies/raw/ml-latest/")

# Load data
movies, ratings, tags = load_all_data(file_dir)

# Build content column
movies = build_content_column(movies)

content = movies["content"].values

# -----------------------------
# Build direct index → title mapping
# -----------------------------
idx_to_movieId = dict(enumerate(movies["movieId"].values))
movieId_to_idx = {mid: idx for idx, mid in idx_to_movieId.items()}

idx_to_title = dict(enumerate(movies["title"].values))
title_to_idx = {title: idx for idx, title in idx_to_title.items()}

item_index = {
    "num_items": len(movies),

    # forward
    "idx_to_movieId": idx_to_movieId,
    "idx_to_title": idx_to_title,

    # reverse
    "movieId_to_idx": movieId_to_idx,
    
    "title_to_idx": title_to_idx,
}

# -----------------------------
# Build user-item matrix
# -----------------------------
user_item_matrix, userId_to_idx, idx_to_userId = build_user_item_matrix(ratings, movies, title_to_idx)

user_index = {
    "num_users": len(userId_to_idx),

    "userId_to_idx": userId_to_idx,
    "idx_to_userId": idx_to_userId,

}

# -----------------------------
# Compute embeddings
# -----------------------------
(
    item_embeddings, 
    sbert_model,
    user_embeddings, 
) = compute_item_and_user_embeddings(content, user_item_matrix)

# -----------------------------
# Build top-k graphs
# -----------------------------
topk_content = build_topk_content(item_embeddings)  # Use FAISS for content topk if needed
topk_cf = build_item_cf_topk(user_item_matrix)

# -----------------------------
# Build FAISS indices
# -----------------------------
faiss_index_item, faiss_index_users = build_faiss_indices(item_embeddings, user_embeddings)

# -----------------------------
# Save artifacts
# -----------------------------


save_pickle(item_index, save_dir.joinpath("item_index.pkl"))
save_pickle(user_index, save_dir.joinpath("user_index.pkl"))

save_numpy(item_embeddings, save_dir.joinpath("movies_item_embeddings.npy"))
save_numpy(user_embeddings, save_dir.joinpath("movies_user_embeddings.npy"))

sbert_model.save(str(save_dir.joinpath("movie_sbert_model")))

save_pickle(topk_content, save_dir.joinpath("movies_item_topk_content.pkl"))
save_pickle(topk_cf, save_dir.joinpath("movies_item_topk_cf.pkl"))

save_npz(save_dir.joinpath("user_item_matrix.npz"), user_item_matrix)

save_faiss_index(faiss_index_users, str(save_dir.joinpath("faiss_index_users.index")))