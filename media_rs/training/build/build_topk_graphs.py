import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix

def build_item_cf_topk(user_item_matrix: csr_matrix, k=100, batch_size=1000):
    num_items = user_item_matrix.shape[1]
    topk_cf = {}

    for start in range(0, num_items, batch_size):
        end = min(start + batch_size, num_items)
        batch_matrix = user_item_matrix[:, start:end].T
        sim = cosine_similarity(batch_matrix, user_item_matrix.T)
        for i, item_idx in enumerate(range(start, end)):
            top_indices = np.argsort(-sim[i])[:k+1]
            topk_cf[item_idx] = [(j, sim[i,j]) for j in top_indices if j != item_idx]
    return topk_cf
