import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, csr_matrix
from typing import Tuple, Dict

def build_user_item_matrix(
    ratings: pd.DataFrame,
    movies: pd.DataFrame,
    title_to_idx: Dict[str, int]
) -> Tuple[csr_matrix, Dict[int,int], Dict[int,int]]:
    """
    Build a user-item sparse matrix using movie titles as columns.

    Args:
        ratings: DataFrame with columns ["userId", "movieId", "rating"]
        movies: DataFrame with columns ["movieId", "title"]
        title_to_idx: Dict mapping normalized movie titles to column indices

    Returns:
        user_item_matrix: CSR sparse matrix of shape (num_users, num_movies)
        userId_to_idx: Dict mapping userId -> row index
        idx_to_userId: Dict mapping row index -> userId
    """

    # Normalize movie titles
    movies = movies.copy()
    movies["title_norm"] = movies["title"].str.strip().str.lower()
    
    # Merge ratings with normalized titles
    ratings = ratings.merge(
        movies[["movieId", "title_norm"]],
        on="movieId",
        how="inner"
    )

    # Keep only movies present in title_to_idx
    ratings = ratings[ratings["title_norm"].isin(title_to_idx.keys())]

    if ratings.empty:
        raise ValueError("No ratings left after filtering with title_to_idx. Check title normalization!")

    # Map users to row indices
    user_ids = ratings["userId"].unique()
    userId_to_idx = {uid: i for i, uid in enumerate(user_ids)}
    idx_to_userId = {i: uid for uid, i in userId_to_idx.items()}

    # Build sparse matrix indices and data
    rows = [userId_to_idx[r.userId] for r in ratings.itertuples(index=False)]
    cols = [title_to_idx[r.title_norm] for r in ratings.itertuples(index=False)]
    data = [r.rating for r in ratings.itertuples(index=False)]

    # Sanity check
    max_col = max(cols)
    if max_col >= len(title_to_idx):
        raise ValueError(f"Column index {max_col} exceeds matrix width {len(title_to_idx)}")

    # Build CSR sparse matrix
    user_item_matrix = coo_matrix(
        (data, (rows, cols)),
        shape=(len(user_ids), len(title_to_idx)),
        dtype=np.float32
    ).tocsr()

    return user_item_matrix, userId_to_idx, idx_to_userId