import numpy as np
from scipy.sparse import save_npz
from media_rs.utils.load_data import save_pickle, save_numpy, save_faiss_index, save_metadata

def save_all_artifacts(wdir, idx_to_movieId, movieId_to_idx, idx_to_userId, userId_to_idx,
                       item_embeddings, user_embeddings, topk_content, topk_cf,
                       user_item_matrix, vectorizer, svd, faiss_index_content, faiss_index_users, movies):

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
