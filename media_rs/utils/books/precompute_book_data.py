from pathlib import Path
from scipy.sparse import save_npz

from media_rs.utils.books.load_book_data import load_all_book_data
from media_rs.utils.books.build_content_features import build_content_column
from media_rs.utils.build_user_item_matrix import build_user_item_matrix
from media_rs.training.features.embeddings import (
    compute_sbert_embeddings, 
    compute_tfidf_embeddings,
    compute_user_embeddings
)
from media_rs.training.build.build_topk_graphs import build_item_cf_topk, build_topk_content
from media_rs.training.build.build_faiss_indices import build_faiss_indices

from media_rs.utils.books.build_item_index import build_book_item_index
from media_rs.utils.load_data import save_pickle, save_numpy, save_faiss_index

save_dir = Path("data/books/cache/")
file_dir = Path("data/books/book_dataset/raw/")

sbert_dir = save_dir / "sbert"
tfidf_dir = save_dir / "tfidf"

sbert_dir.mkdir(parents=True, exist_ok=True)
tfidf_dir.mkdir(parents=True, exist_ok=True)

# Load data
books, ratings, tags = load_all_book_data(file_dir)

# Build content column
books = build_content_column(books)

content = books["content"].values

# build item index
item_index = build_book_item_index(books)

# -----------------------------
# Build user-item matrix
# -----------------------------
user_item_matrix, userId_to_idx, idx_to_userId = build_user_item_matrix(
    ratings, 
    item_index["itemId_to_idx"]
)

user_index = {
    "num_users": len(userId_to_idx),

    "userId_to_idx": userId_to_idx,
    "idx_to_userId": idx_to_userId,

}

# -----------------------------
# SBERT embeddings
# -----------------------------
sbert_model, sbert_item_embeddings = compute_sbert_embeddings(content)

sbert_user_embeddings = compute_user_embeddings(
    user_item_matrix,
    sbert_item_embeddings,
)

# -----------------------------
# TF-IDF embeddings
# -----------------------------
tfidf_item_embeddings, vectorizer, svd = compute_tfidf_embeddings(content)

tfidf_user_embeddings = compute_user_embeddings(
    user_item_matrix,
    tfidf_item_embeddings,
)

# Content-based
topk_content_sbert = build_topk_content(sbert_item_embeddings)
topk_content_tfidf = build_topk_content(tfidf_item_embeddings)

# Collaborative filtering (shared)
topk_cf = build_item_cf_topk(user_item_matrix)

# -----------------------------
# Build FAISS indices
# -----------------------------
faiss_item_sbert, faiss_users_sbert = build_faiss_indices(sbert_item_embeddings, sbert_user_embeddings)
faiss_item_tfidf, faiss_users_tfidf = build_faiss_indices(tfidf_item_embeddings, tfidf_user_embeddings)

# -----------------------------
# Save artifacts
# -----------------------------


save_pickle(item_index, save_dir.joinpath("item_index.pkl"))
save_pickle(user_index, save_dir.joinpath("user_index.pkl"))

save_numpy(tfidf_item_embeddings, tfidf_dir.joinpath("item_embeddings.npy"))
save_numpy(tfidf_user_embeddings, tfidf_dir.joinpath("user_embeddings.npy"))

save_numpy(sbert_item_embeddings, sbert_dir.joinpath("item_embeddings.npy"))
save_numpy(sbert_user_embeddings, sbert_dir.joinpath("user_embeddings.npy"))

sbert_model.save(str(sbert_dir.joinpath("sbert_model")))


save_pickle(topk_content_tfidf, tfidf_dir.joinpath("item_topk_content.pkl"))
save_pickle(topk_content_sbert, sbert_dir.joinpath("item_topk_content.pkl"))

save_pickle(topk_cf, save_dir.joinpath("item_topk_cf.pkl"))

save_npz(save_dir.joinpath("user_item_matrix.npz"), user_item_matrix)

save_faiss_index(faiss_users_tfidf, str(tfidf_dir.joinpath("faiss_index_users.index")))
save_faiss_index(faiss_users_sbert, str(sbert_dir.joinpath("faiss_index_users.index")))

save_pickle(vectorizer, tfidf_dir / "tfidf_vectorizer.pkl")
save_pickle(svd, tfidf_dir / "svd.pkl")

save_faiss_index(
    faiss_item_sbert,
    str(sbert_dir / "faiss_index_items.index"),
)

save_faiss_index(
    faiss_item_tfidf,
    str(tfidf_dir / "faiss_index_items.index"),
)