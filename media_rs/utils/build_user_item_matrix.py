import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, csr_matrix
from typing import Tuple, Dict

from media_rs.utils.load_data import norm

def build_user_item_matrix(
    ratings: pd.DataFrame,
    itemId_to_idx: Dict[int, int]
):
    ratings = ratings[ratings["itemId"].isin(itemId_to_idx)]

    if ratings.empty:
        raise ValueError("No ratings left after filtering with itemId_to_idx")

    user_ids = ratings["userId"].unique()
    userId_to_idx = {uid: i for i, uid in enumerate(user_ids)}
    idx_to_userId = {i: uid for uid, i in userId_to_idx.items()}

    rows = [userId_to_idx[r.userId] for r in ratings.itertuples(index=False)]
    cols = [itemId_to_idx[r.itemId] for r in ratings.itertuples(index=False)]
    data = [r.rating for r in ratings.itertuples(index=False)]

    user_item_matrix = coo_matrix(
        (data, (rows, cols)),
        shape=(len(user_ids), len(itemId_to_idx)),
        dtype=np.float32
    ).tocsr()

    return user_item_matrix, userId_to_idx, idx_to_userId
