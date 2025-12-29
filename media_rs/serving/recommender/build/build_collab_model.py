import numpy as np
import pickle
from pathlib import Path

from scipy.sparse import load_npz

from media_rs.serving.recommender.models.collab import (
    ItemItemCollaborativeModel,
    UserCollaborativeModel,
)
from media_rs.utils.item_index import ItemIndex
from media_rs.training.features.faiss import load_faiss_index

def get_item_cf_model(item_index: ItemIndex):
    wdir = Path("media_rs/serving/artifacts")
    
    item_cf_topk_path = wdir.joinpath("movies_item_topk_cf.pkl")
    
    with open(item_cf_topk_path, "rb") as f:
        topk_graph_content = pickle.load(f)
    
    return ItemItemCollaborativeModel(
        ids=item_index.ids,
        topk_graph=topk_graph_content
    )
    
def get_user_cf_model():
    wdir = Path("media_rs/serving/artifacts")
    
    faiss_index_path = wdir.joinpath("faiss_index_users.index")
    user_item_matrix_path = wdir.joinpath("user_item_matrix.npz")
    item_embeddings_path = wdir.joinpath("movies_item_embeddings.npy")
    
    faiss_index = load_faiss_index(str(faiss_index_path))
    user_item_matrix = load_npz(str(user_item_matrix_path))
    item_embeddings = np.load(item_embeddings_path)
    
    return UserCollaborativeModel(
        faiss_index=faiss_index,
        user_item_matrix=user_item_matrix,
        item_embeddings=item_embeddings
    )
    