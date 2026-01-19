from pathlib import Path
from scipy.sparse import save_npz

from media_rs.utils.movies.load_movie_data import load_all_movie_data
from media_rs.utils.movies.build_content_features import build_content_column
from media_rs.utils.movies.build_user_item_matrix import build_user_item_matrix
from media_rs.training.build.compute_embeddings import compute_item_and_user_embeddings
from media_rs.training.build.build_topk_graphs import build_item_cf_topk, build_topk_content
from media_rs.training.build.build_faiss_indices import build_faiss_indices

from media_rs.utils.movies.build_item_index import build_item_index
from media_rs.utils.load_data import save_pickle, save_numpy, save_faiss_index

save_dir = Path("data/movies/cache/")
file_dir = Path("data/movies/raw/ml-latest/")
# save_dir = Path("data/movies/raw/ml-latest-small/cache/")
# file_dir = Path("data/movies/raw/ml-latest-small/")

# Load data
movies, ratings, tags, links = load_all_movie_data(file_dir)

# Build content column
movies = build_content_column(movies)

content = movies["content"].values

# build item index
item_index = build_item_index(movies, links)

# -----------------------------
# Build user-item matrix
# -----------------------------
user_item_matrix, userId_to_idx, idx_to_userId = build_user_item_matrix(
    ratings, 
    movies, 
    item_index["movieId_to_idx"]
)

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