import numpy as np
import faiss

from media_rs.features.faiss import (
    build_faiss_index, 
    query_faiss_topk, 
    save_faiss_index
)

from media_rs.utils.load_data import (
    load_dataframe,
    add_tags_to_movies,
    save_numpy,
    save_pickle,
    save_metadata
)

from media_rs.features.embeddings import (
    compute_item_embeddings,
    compute_user_embeddings,
    compute_topk_similarity_graph
)

# Load CSVs
movies = load_dataframe('data/movies/raw/movies.csv')
ratings = load_dataframe('data/movies/raw/ratings.csv')
tags = load_dataframe('data/movies/raw/tags.csv')

# Combine genres + tags (if not already done)
movies = add_tags_to_movies(movies, tags)

movies["content"] = (
    movies[["genres", "tag"]]
    .fillna("")
    .agg(" ".join, axis=1)
)
movie_texts = movies['content'].values

# 1. Compute item embeddings
item_embeddings, vectorizer, svd = compute_item_embeddings(movie_texts)

# 2. Create user-item dictionary
user_item_dict = ratings.groupby("userId")["movieId"].apply(list).to_dict()

movieId_to_idx = {mid: idx for idx, mid in enumerate(movies['movieId'].values)}
user_item_dict_mapped = {uid: [movieId_to_idx[mid] for mid in mids if mid in movieId_to_idx] 
                         for uid, mids in user_item_dict.items()}

# 3. Compute user embeddings
user_embeddings = compute_user_embeddings(user_item_dict_mapped, item_embeddings)

# 1. Build FAISS index
index = build_faiss_index(item_embeddings, metric="cosine")

# 2. Query top-K neighbors for all items
topk_items = query_faiss_topk(index, item_embeddings, k=100)

# 5. Save everything
save_numpy(item_embeddings, "data/movies/movies_item_embeddings.npy")
save_numpy(np.array(list(user_embeddings.values())), "data/movies/movies_user_embeddings.npy")
save_pickle(topk_items, "data/movies/movies_item_topk.pkl")
save_pickle(vectorizer, "data/movies/movies_vectorizer.pkl")
save_pickle(svd, "data/movies/movies_svd.pkl")
save_metadata(movies, "data/movies/movies_item_metadata.parquet")
