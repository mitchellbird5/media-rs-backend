from pathlib import Path
from media_rs.training.build.load_data import load_all_data
from media_rs.training.build.build_content_features import build_content_column
from media_rs.training.build.build_user_item_matrix import build_user_item_matrix
from media_rs.training.build.compute_embeddings import compute_item_and_user_embeddings
from media_rs.training.build.build_topk_graphs import build_item_cf_topk
from media_rs.training.build.build_faiss_indices import build_faiss_indices
from media_rs.training.build.save_artifacts import save_all_artifacts

wdir = Path("media_rs/serving/artifacts/")

# Load data
movies, ratings, tags = load_all_data(Path("data/movies/raw/ml-latest/"))

# Build content column
movies = build_content_column(movies)

# Stable mapping
movie_ids = movies["movieId"].values
idx_to_movieId = dict(enumerate(movie_ids))
movieId_to_idx = {mid: idx for idx, mid in idx_to_movieId.items()}

# Build user-item matrix
user_item_matrix, userId_to_idx, idx_to_userId = build_user_item_matrix(ratings, movieId_to_idx)

# Compute embeddings
item_embeddings, user_embeddings, vectorizer, svd = compute_item_and_user_embeddings(movies, user_item_matrix)

# Build top-k graphs
topk_content = None  # Use FAISS for content topk if needed
topk_cf = build_item_cf_topk(user_item_matrix)

# Build FAISS indices
faiss_index_content, faiss_index_users = build_faiss_indices(item_embeddings, user_embeddings)

# Save artifacts
save_all_artifacts(
    wdir, idx_to_movieId, movieId_to_idx, idx_to_userId, userId_to_idx,
    item_embeddings, user_embeddings, topk_content, topk_cf,
    user_item_matrix, vectorizer, svd, faiss_index_content, faiss_index_users, movies
)
