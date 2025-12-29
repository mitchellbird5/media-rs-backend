import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from typing import Tuple, Dict

def build_user_item_matrix(ratings, movieId_to_idx) -> Tuple[csr_matrix, Dict[int,int], Dict[int,int]]:
    ratings = ratings[ratings["movieId"].isin(movieId_to_idx)]
    user_ids = ratings["userId"].unique()
    userId_to_idx = {uid: i for i, uid in enumerate(user_ids)}
    idx_to_userId = {i: uid for uid, i in userId_to_idx.items()}

    rows = [userId_to_idx[r.userId] for r in ratings.itertuples(index=False)]
    cols = [movieId_to_idx[r.movieId] for r in ratings.itertuples(index=False)]
    data = [r.rating for r in ratings.itertuples(index=False)]

    user_item_matrix = coo_matrix((data, (rows, cols)),
                                  shape=(len(user_ids), len(movieId_to_idx)),
                                  dtype=np.float32).tocsr()
    return user_item_matrix, userId_to_idx, idx_to_userId
