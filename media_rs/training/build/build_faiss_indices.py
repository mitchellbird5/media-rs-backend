import faiss
from media_rs.training.features.faiss import build_faiss_index

def build_faiss_indices(item_embeddings, user_embeddings):
    faiss_index_content = build_faiss_index(item_embeddings, metric="cosine")

    faiss.normalize_L2(user_embeddings)
    faiss_index_users = faiss.IndexFlatIP(user_embeddings.shape[1])
    faiss_index_users.add(user_embeddings)

    return faiss_index_content, faiss_index_users
